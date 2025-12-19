# model.py
# DM-CER: Dynamic Multimodal Causal Emotion Reasoner
# Compatible con preprocesado MELD → JSONL (text_emb, audio, vision, speaker_idx, label)
# PyTorch >= 1.12

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilidades
# ----------------------------

def _gelu(x):
    return F.gelu(x)


def _mlp(d_in, d_hidden, d_out, p=0.1):
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(d_hidden, d_out),
    )


# ----------------------------
# Encoders por modalidad
# ----------------------------

class ModalityEncoder(nn.Module):
    """Proyección + normalización a espacio común d_model."""
    def __init__(self, d_in: int, d_model: int, p: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in)
        h = self.proj(x)
        h = _gelu(h)
        h = self.drop(self.norm(h))
        return h  # (B,T,d_model)


# ----------------------------
# Memoria por hablante
# ----------------------------

class SpeakerMemory(nn.Module):
    """
    Memoria GRU por hablante; sin in-place sobre el banco `mem`.
    Compatible con AMP: dtypes consistentes en scatter_ y paso GRU en dtype nativo.
    """
    def __init__(self, d_in: int, d_mem: int, speakers_max: int = 10):
        super().__init__()
        self.gru = nn.GRU(d_in, d_mem, batch_first=True)
        self.d_mem = d_mem
        self.speakers_max = speakers_max

    def forward(self, x_in: torch.Tensor, speakers: torch.Tensor) -> torch.Tensor:
        B, T, D = x_in.shape
        device = x_in.device
        dtype_in = x_in.dtype  # puede ser fp16 bajo autocast

        # Bancos con mismo dtype que la entrada (no requieren grad)
        mem = torch.zeros(B, self.speakers_max, self.d_mem, device=device, dtype=dtype_in)
        out = torch.zeros(B, T, self.d_mem, device=device, dtype=dtype_in)

        for t in range(T):
            sidx = speakers[:, t].clamp(min=0, max=self.speakers_max - 1)   # (B,)
            xt = x_in[:, t, :].unsqueeze(1)                                 # (B,1,D)

            # Lee el estado actual del hablante (clon para no compartir storage en el grafo)
            mt = torch.gather(mem, 1, sidx.view(B,1,1).expand(B,1,self.d_mem)).clone()  # (B,1,d_mem)

            # GRU en dtype nativo de sus pesos (normalmente fp32), fuera de autocast
            gru_dtype = self.gru.weight_ih_l0.dtype
            with torch.cuda.amp.autocast(enabled=False):
                xt_g = xt.to(gru_dtype)
                h0_g = mt.transpose(0, 1).to(gru_dtype).contiguous()
                _, new_state = self.gru(xt_g, h0_g)   # (1,B,d_mem) en gru_dtype

            new_state = new_state.transpose(0, 1)     # (B,1,d_mem)
            new_state = new_state.to(dtype_in)        # iguala dtype al banco

            # 🔁 Crear un nuevo banco y actualizar (sin in-place sobre el anterior)
            mem_next = mem.clone()
            mem_next.scatter_(1, sidx.view(B,1,1).expand(B,1,self.d_mem), new_state)
            mem = mem_next

            out[:, t, :] = new_state.squeeze(1)

        return out  # (B,T,d_mem)




# ----------------------------
# Grafo causal dinámico (message passing atencional con ventana temporal)
# ----------------------------

class DynamicCausalGNN(nn.Module):
    """
    Message passing atencional con ventana temporal ±W.
    Compatible con versiones de PyTorch que usan `src_mask=` o `mask=`.
    """
    def __init__(self, d_model: int, layers: int = 2, heads: int = 4, p: float = 0.1, causal_window: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=4 * d_model,
                dropout=p,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(layers)
        ])
        self.causal_window = causal_window

    def _temporal_bool_mask(self, T: int, device) -> torch.Tensor:
        # True donde NO se puede atender (fuera de la ventana)
        idx = torch.arange(T, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()  # (T,T)
        return dist > self.causal_window  # bool

    def _to_additive_mask(self, bool_mask: torch.Tensor) -> torch.Tensor:
        # Convierte bool -> máscara aditiva (0.0 en permitido, -inf en prohibido)
        add = torch.zeros_like(bool_mask, dtype=torch.float32)
        add = add.masked_fill(bool_mask, float("-inf"))
        return add

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B,T,D)
        key_padding_mask (opcional): (B,T) con True en posiciones de PAD (no atender ahí).
        """
        B, T, D = x.shape
        device = x.device

        # Máscara temporal (T,T)
        bool_mask = self._temporal_bool_mask(T, device)              # bool
        add_mask  = self._to_additive_mask(bool_mask)                # float(-inf/0)

        h = x
        for layer in self.layers:
            # Algunas versiones usan src_mask/src_key_padding_mask,
            # otras usan mask/key_padding_mask. Probamos ambas.
            try:
                h = layer(h, src_mask=add_mask, src_key_padding_mask=key_padding_mask)
            except TypeError:
                # Fallback a firmas nuevas
                h = layer(h, mask=add_mask, key_padding_mask=key_padding_mask)
        return h


# ----------------------------
# VAE cross-modal (T+V → A)
# ----------------------------

class CrossModalVAE(nn.Module):
    """
    Reconstruye características de audio a partir de texto+visión.
    Devuelve: recon (B,T,d_audio), kld (scalar), recon_loss (scalar opcional)
    """
    def __init__(self, d_text: int, d_vision: int, d_audio: int, d_latent: int = 64, p: float = 0.1):
        super().__init__()
        d_in = d_text + d_vision
        self.enc = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 2 * d_latent)
        )
        self.dec = nn.Sequential(
            nn.Linear(d_latent, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, d_audio)
        )
        self.d_latent = d_latent

    def forward(self, t: torch.Tensor, v: torch.Tensor, target_audio: Optional[torch.Tensor] = None):
        # t,v,target_audio: (B,T,dt/dv/da)
        x = torch.cat([t, v], dim=-1)  # (B,T,dt+dv)
        stats = self.enc(x)
        mu, logvar = stats.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps  # reparam

        recon = self.dec(z)  # (B,T,d_audio)

        # KL divergence (promedio sobre batch y tiempo)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        recon_loss = None
        if target_audio is not None:
            recon_loss = F.l1_loss(recon, target_audio)

        return recon, kld, recon_loss


# ----------------------------
# Modelo principal: DM-CER
# ----------------------------

@dataclass
class DMConfig:
    # Dimensiones de entrada
    d_text: int = 768
    d_audio: int = 128
    # d_vision: int = 512
    d_vision: int = 768

    # Espacios internos
    d_model: int = 256
    d_mem: int = 128
    num_classes: int = 7

    # GNN
    gnn_layers: int = 2
    gnn_heads: int = 4
    gnn_dropout: float = 0.1
    causal_window: int = 6

    # VAE
    d_latent: int = 64
    vae_dropout: float = 0.1

    # Regularización
    enc_dropout: float = 0.1
    fuse_dropout: float = 0.1

    # Missing modality dropout (sólo en train)
    missing_modality_p: float = 0.2


class DM_CER(nn.Module):
    """
    Dynamic Multimodal Causal Emotion Reasoner.
    Forward espera un batch con:
      - "text":   (B,T,d_text)
      - "audio":  (B,T,d_audio)
      - "vision": (B,T,d_vision)
      - "speakers": (B,T) int64
      - "length": (B,) int64 (opcional; no estrictamente necesario aquí)

    Devuelve un dict con:
      - logits: (B,T,C) cabeza fusionada
      - logits_t / logits_a / logits_v: (B,T,C) por modalidad
      - recon_audio: (B,T,d_audio), kld (scalar), recon_loss (scalar opcional)
    """
    def __init__(self, cfg: Optional[DMConfig] = None):
        super().__init__()
        self.cfg = cfg or DMConfig()
        D = self.cfg.d_model

        # Encoders
        self.text_enc = ModalityEncoder(self.cfg.d_text, D, p=self.cfg.enc_dropout)
        self.audio_enc = ModalityEncoder(self.cfg.d_audio, D, p=self.cfg.enc_dropout)
        self.vision_enc = ModalityEncoder(self.cfg.d_vision, D, p=self.cfg.enc_dropout)

        # Fusión temprana + proyección
        self.fuse = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.ReLU(),
            nn.Dropout(self.cfg.fuse_dropout),
            nn.LayerNorm(D)
        )

        # Memoria por hablante (sobre representación fusionada)
        self.mem = SpeakerMemory(D, self.cfg.d_mem)

        # GNN causal sobre [fused ⊕ memory]
        self.gnn = DynamicCausalGNN(
            d_model=D + self.cfg.d_mem,
            layers=self.cfg.gnn_layers,
            heads=self.cfg.gnn_heads,
            p=self.cfg.gnn_dropout,
            causal_window=self.cfg.causal_window
        )

        # Cabezas por modalidad y fusionada
        self.head_text = nn.Linear(D, self.cfg.num_classes)
        self.head_audio = nn.Linear(D, self.cfg.num_classes)
        self.head_vis = nn.Linear(D, self.cfg.num_classes)
        self.head_fused = nn.Linear(D + self.cfg.d_mem, self.cfg.num_classes)

        # VAE (T+V → A)
        self.vae = CrossModalVAE(D, D, self.cfg.d_audio, d_latent=self.cfg.d_latent, p=self.cfg.vae_dropout)

    def _apply_missing_modality_dropout(self, x: torch.Tensor, p: float) -> torch.Tensor:
        if p <= 0.0 or not self.training:
            return x
        # Apaga toda la representación del turno con prob. p (canal completo)
        mask = (torch.rand_like(x[..., :1]) > p).float()
        return x * mask

    def forward(self, batch: Dict[str, torch.Tensor], missing_modality_p: Optional[float] = None) -> Dict[str, Any]:
        """
        batch: diccionario con tensores (ver docstring de clase).
        missing_modality_p: si se pasa, sobreescribe cfg.missing_modality_p.
        """
        p_mm = self.cfg.missing_modality_p if missing_modality_p is None else missing_modality_p

        # 1) Encoders por modalidad
        x_t = self.text_enc(batch["text"])    # (B,T,D)
        x_a = self.audio_enc(batch["audio"])  # (B,T,D)
        x_v = self.vision_enc(batch["vision"])# (B,T,D)

        # 2) Missing-modality dropout (sólo entrenamiento)
        x_t = self._apply_missing_modality_dropout(x_t, p_mm)
        x_a = self._apply_missing_modality_dropout(x_a, p_mm)
        x_v = self._apply_missing_modality_dropout(x_v, p_mm)

        # 3) Fusión temprana
        fused = self.fuse(torch.cat([x_t, x_a, x_v], dim=-1))  # (B,T,D)

        # 4) Memoria por hablante
        mem = self.mem(fused, batch["speakers"])               # (B,T,d_mem)

        # 5) Grafo causal dinámico sobre [fused ⊕ mem]
        gnn_in = torch.cat([fused, mem], dim=-1)               # (B,T,D+d_mem)
        h = self.gnn(gnn_in)                                   # (B,T,D+d_mem)

        # 6) Cabezas de clasificación
        logits_fused = self.head_fused(h)      # (B,T,C)
        logits_t = self.head_text(x_t)         # (B,T,C)
        logits_a = self.head_audio(x_a)        # (B,T,C)
        logits_v = self.head_vis(x_v)          # (B,T,C)

        # 7) VAE (reconstrucción de audio desde T+V)
        recon_audio, kld, recon_loss = self.vae(x_t, x_v, target_audio=batch.get("audio", None))

        return {
            "logits": logits_fused,
            "logits_t": logits_t,
            "logits_a": logits_a,
            "logits_v": logits_v,
            "recon_audio": recon_audio,
            "kld": kld,
            "recon_loss": recon_loss,
        }


# ----------------------------
# Pérdidas auxiliares (opcionales)
# ----------------------------

def seq_ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Cross-entropy por turno con enmascarado de padding.
    logits: (B,T,C), labels: (B,T)
    """
    B, T, C = logits.shape
    return F.cross_entropy(logits.reshape(B * T, C), labels.reshape(B * T), ignore_index=ignore_index)


def kl_div_logits(p_logits: torch.Tensor, q_logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """KL(p || q) entre distribuciones softmax con temperatura T."""
    p = F.log_softmax(p_logits / T, dim=-1)
    q = F.softmax(q_logits / T, dim=-1)
    return F.kl_div(p, q, reduction="batchmean") * (T * T)


def consistency_loss(logits_t: torch.Tensor, logits_a: torch.Tensor, logits_v: torch.Tensor, logits_fused: torch.Tensor) -> torch.Tensor:
    """Promedia KL en ambas direcciones entre cada modalidad y la cabeza fusionada."""
    L = 0.0
    for x in (logits_t, logits_a, logits_v):
        L = L + kl_div_logits(x, logits_fused) + kl_div_logits(logits_fused, x)
    return L / 6.0


# ----------------------------
# Ejemplo de uso
# ----------------------------
if __name__ == "__main__":
    # Demo rápido con tensores dummy
    B, T = 2, 5
    cfg = DMConfig()
    model = DM_CER(cfg)

    batch = {
        "text": torch.randn(B, T, cfg.d_text),
        "audio": torch.randn(B, T, cfg.d_audio),
        "vision": torch.randn(B, T, cfg.d_vision),
        "speakers": torch.randint(0, 4, (B, T)),
        "length": torch.full((B,), T, dtype=torch.long),
        "labels": torch.randint(0, cfg.num_classes, (B, T))
    }

    out = model(batch)
    loss_cls = seq_ce_loss(out["logits"], batch["labels"])
    loss_cons = consistency_loss(out["logits_t"], out["logits_a"], out["logits_v"], out["logits"])
    loss_recon = out["recon_loss"] if out["recon_loss"] is not None else torch.tensor(0.0)
    loss_kld = out["kld"]

    total = 1.0 * loss_cls + 0.3 * loss_cons + 0.2 * (loss_recon + 0.1 * loss_kld)
    print(f"demo loss: {total.item():.4f}")
