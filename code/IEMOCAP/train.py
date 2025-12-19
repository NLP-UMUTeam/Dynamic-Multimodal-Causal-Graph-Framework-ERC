import os, json, argparse, math, random
from typing import Dict, List, Any
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from model import DM_CER, DMConfig, seq_ce_loss, consistency_loss


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # o ":4096:8" antes de importar torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------
# Utilidades
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # True suele ir mejor para vídeo/audio

def class_weights_from_data(jsonl_path: str, num_classes: int) -> torch.Tensor:
    """Calcula pesos inversamente proporcionales a la frecuencia (por turno)."""
    if not os.path.isfile(jsonl_path):  # dev/test opcional
        return torch.ones(num_classes)
    cnt = np.zeros(num_classes, dtype=np.int64)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for t in d["turns"]:
                y = int(t["label"])
                if 0 <= y < num_classes:
                    cnt[y] += 1
    cnt = np.maximum(cnt, 1)
    w = cnt.sum() / (cnt.astype(np.float32) * num_classes)
    return torch.tensor(w, dtype=torch.float32)

def load_config_override(path: str, cfg: DMConfig) -> DMConfig:
    p = os.path.join(path, "dmcer_config_override.json")
    if os.path.isfile(p):
        with open(p, "r") as f:
            o = json.load(f)
        for k,v in o.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


# ---------------------------
# Dataset y DataLoader
# ---------------------------

class IEMOCAPJSONL(Dataset):
    """
    Cada línea:
    {
      "dialog_id": int,
      "speakers": ["A","B",...],
      "turns": [
        {"speaker_idx":0,"text_emb":[...d_text...],"audio":[...d_audio...],
         "vision":[...d_vision...],"label":int, "dialogue_id":int, "utterance_id":int},
        ...
      ]
    }
    """
    def __init__(self, jsonl_path: str, d_text: int, d_audio: int, d_vision: int):
        assert os.path.isfile(jsonl_path), f"No existe {jsonl_path}"
        self.path = jsonl_path
        self.d_text, self.d_audio, self.d_vision = d_text, d_audio, d_vision
        self.samples: List[Dict[str,Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        turns = s["turns"]
        # Convertimos a tensores por turno (T variable)
        text   = torch.tensor([t["text_emb"] for t in turns], dtype=torch.float32)   # (T,d_text)
        audio  = torch.tensor([t["audio"]     for t in turns], dtype=torch.float32)  # (T,d_audio)
        vision = torch.tensor([t["vision"]    for t in turns], dtype=torch.float32)  # (T,d_vision)
        labels = torch.tensor([t["label"]     for t in turns], dtype=torch.long)     # (T,)
        spk    = torch.tensor([t["speaker_idx"] for t in turns], dtype=torch.long)   # (T,)
        return {
            "text": text, "audio": audio, "vision": vision,
            "labels": labels, "speakers": spk, "length": torch.tensor(len(turns), dtype=torch.long),
            "dialog_id": s.get("dialog_id", -1),
        }

def pad_sequence_2d(seqs: List[torch.Tensor], dim: int) -> torch.Tensor:
    # seqs: list of (T,dim) -> (B, Tmax, dim)
    Tmax = max(x.size(0) for x in seqs)
    B = len(seqs)
    out = torch.zeros(B, Tmax, dim, dtype=seqs[0].dtype)
    for i,x in enumerate(seqs):
        T = x.size(0)
        out[i, :T, :] = x
    return out

def collate_batch(batch: List[Dict[str,Any]]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    lengths = torch.stack([b["length"] for b in batch])  # (B,)
    Tmax = int(lengths.max().item())

    text   = pad_sequence_2d([b["text"] for b in batch],   batch[0]["text"].size(1))
    audio  = pad_sequence_2d([b["audio"] for b in batch],  batch[0]["audio"].size(1))
    vision = pad_sequence_2d([b["vision"] for b in batch], batch[0]["vision"].size(1))

    labels = torch.full((B, Tmax), -100, dtype=torch.long)
    speakers = torch.zeros(B, Tmax, dtype=torch.long)
    for i,b in enumerate(batch):
        T = b["length"].item()
        labels[i, :T] = b["labels"]
        speakers[i, :T] = b["speakers"]

    return {
        "text": text, "audio": audio, "vision": vision,
        "labels": labels, "speakers": speakers, "length": lengths
    }


# ---------------------------
# Entrenamiento / evaluación
# ---------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    all_y, all_p = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outs = model(batch, missing_modality_p=0.0)
        logits = outs["logits"]                      # (B,T,C)
        labels = batch["labels"]                     # (B,T)
        mask = labels != -100
        if mask.sum() == 0:
            continue
        y = labels[mask].detach().cpu().numpy()
        p = logits[mask].argmax(-1).detach().cpu().numpy()
        all_y.extend(y.tolist()); all_p.extend(p.tolist())
    acc = accuracy_score(all_y, all_p) if all_y else 0.0
    f1w = f1_score(all_y, all_p, average="weighted") if all_y else 0.0
    f1m = f1_score(all_y, all_p, average="macro") if all_y else 0.0
    return {"acc": acc, "f1_weighted": f1w, "f1_macro": f1m}

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, scaler, device: str,
                    ce_loss_fn, w_consistency: float) -> float:
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=False):
            outs = model(batch)  # usa p de missing-modality del cfg
            L_cls  = ce_loss_fn(outs["logits"], batch["labels"])
            L_cons = consistency_loss(outs["logits_t"], outs["logits_a"], outs["logits_v"], outs["logits"])
            L_rec  = outs["recon_loss"] if outs["recon_loss"] is not None else torch.tensor(0.0, device=device)
            L_kld  = outs["kld"]
            loss = L_cls + w_consistency * L_cons + 0.2 * (L_rec + 0.1 * L_kld)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().item())
    return total / max(1, len(loader))

def save_checkpoint(state: Dict[str,Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Carpeta con train.jsonl, dev.jsonl, test.jsonl")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--weight_decay", type=float, default=0.0005)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--w_consistency", type=float, default=0.03)
    ap.add_argument("--early_patience", type=int, default=10)
    ap.add_argument("--resume", type=str, default="", help="Ruta a checkpoint .pt para reanudar")
    ap.add_argument("--save_dir", type=str, default="./checkpoints_dmcer")
    args = ap.parse_args()

    set_seed(args.seed)

    # Config del modelo (ajusta d_... con override del preprocesado si existe)
    cfg = DMConfig()
    cfg = load_config_override(args.data_root, cfg)

    # Datasets / Loaders
    train_path = os.path.join(args.data_root, "iemocap_train.jsonl")
    # dev_path   = os.path.join(args.data_root, "dev.jsonl")
    test_path  = os.path.join(args.data_root, "iemocap_test.jsonl")

    ds_tr = IEMOCAPJSONL(train_path, cfg.d_text, cfg.d_audio, cfg.d_vision)
    
    dialogs = [s.get("dialog_id", i) for i, s in enumerate(ds_tr.samples)]
    unique_dialogs = list(dict.fromkeys(dialogs))  # preserva orden
    rng = random.Random(args.seed)
    rng.shuffle(unique_dialogs)
    n_val = max(1, int(round(0.10 * len(unique_dialogs))))
    val_dialogs = set(unique_dialogs[:n_val])
    tr_indices, va_indices = [], []
    for i, d in enumerate(dialogs):
        if d in val_dialogs:
            va_indices.append(i)
        else:
            tr_indices.append(i)
    ds_va = Subset(ds_tr, va_indices)
        
        
    ds_te = IEMOCAPJSONL(test_path,  cfg.d_text, cfg.d_audio, cfg.d_vision) if os.path.isfile(test_path) else None

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch) if ds_va else None
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch) if ds_te else None

    # Modelo
    device = args.device
    model = DM_CER(cfg).to(device)

    # Pesos de clase (opcional; mejora F1 macro si hay desbalance)
    class_w = class_weights_from_data(train_path, cfg.num_classes).to(device)
    def ce_loss_fn(logits, labels):
        B,T,C = logits.size()
        return nn.CrossEntropyLoss(weight=class_w, ignore_index=-100)(logits.view(B*T, C), labels.view(B*T))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_f1 = -1.0

    # Reanudar si corresponde
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", best_f1)
        print(f"Reanudado desde {args.resume} (epoch {ckpt['epoch']})")

    # Entrenamiento
    patience = args.early_patience
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

    for ep in range(start_epoch, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_tr, optimizer, scaler, device, ce_loss_fn, args.w_consistency)
        scheduler.step()

        if dl_va:
            metrics = evaluate(model, dl_va, device)
            print(f"Epoch {ep:03d} | train loss {tr_loss:.4f} | "
                  f"val acc {metrics['acc']:.4f} | f1_w {metrics['f1_weighted']:.4f} | f1_m {metrics['f1_macro']:.4f}")
            f1w = metrics["f1_weighted"]
        else:
            print(f"Epoch {ep:03d} | train loss {tr_loss:.4f}")
            f1w = tr_loss * -1  # para que siempre mejore si no hay val

        # Guardar "last"
        save_checkpoint({
            "epoch": ep,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": asdict(cfg),
            "best_f1": best_f1
        }, last_path)

        # Early stopping + "best"
        improved = f1w > best_f1
        if improved:
            best_f1 = f1w
            save_checkpoint({
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": asdict(cfg),
                "best_f1": best_f1
            }, best_path)
            patience = args.early_patience
            print(f"✔️ Nuevo mejor (F1w={best_f1:.4f}). Guardado en {best_path}")
        # else:
        #     patience -= 1
        #     if patience <= 0:
        #         print("Early stopping por paciencia agotada.")
        #         break

    # Evaluación en test con el mejor modelo
    if dl_te:
        print("Cargando mejor checkpoint y evaluando en test…")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, dl_te, device)
        print(f"TEST | acc {test_metrics['acc']:.4f} | f1_w {test_metrics['f1_weighted']:.4f} | f1_m {test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
