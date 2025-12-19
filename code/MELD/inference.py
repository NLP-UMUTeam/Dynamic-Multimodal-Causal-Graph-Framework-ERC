# infer.py
# Inferencia en MELD test para DM-CER -> CSV con columna 'prediction'
# Uso:
#   python infer.py \
#     --data_root /data/ronghao/IEMOCAP/MELD/DM-CER-processed \
#     --ckpt ./checkpoints_dmcer/best.pt \
#     --out_csv ./pred_test.csv \
#     [--csv_in /data/ronghao/IEMOCAP/MELD/MELD.Raw/test/test_sent_emo.csv] \
#     [--batch_size 8 --device cuda]

import os, json, argparse
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd

from model import DM_CER, DMConfig

EMO_ID2NAME = {0:"neutral", 1:"joy", 2:"sadness", 3:"anger", 4:"surprise", 5:"fear", 6:"disgust"}

# ---------------------------
# Dataset con IDs por turno
# ---------------------------

class MELDJSONLWithIDs(Dataset):
    """
    Cada línea del JSONL:
    {
      "dialog_id": int,
      "speakers": ["A","B",...],
      "turns": [
        {"speaker_idx": int, "text_emb": [...], "audio": [...], "vision": [...],
         "label": int, "dialogue_id": int, "utterance_id": int},
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
        T = len(turns)
        text   = torch.tensor([t["text_emb"] for t in turns], dtype=torch.float32)    # (T,d_text)
        audio  = torch.tensor([t["audio"]     for t in turns], dtype=torch.float32)   # (T,d_audio)
        vision = torch.tensor([t["vision"]    for t in turns], dtype=torch.float32)   # (T,d_vision)
        labels = torch.tensor([t.get("label", -100) for t in turns], dtype=torch.long)  # (T,)
        spk    = torch.tensor([t["speaker_idx"] for t in turns], dtype=torch.long)    # (T,)
        dia_ids = torch.tensor(
            [t.get("dialogue_id", s.get("dialog_id", -1)) for t in turns],
            dtype=torch.long
        )
        utt_ids = torch.tensor(
            [t.get("utterance_id", i+1) for i, t in enumerate(turns)],
            dtype=torch.long
        )

        return {
            "text": text, "audio": audio, "vision": vision,
            "labels": labels, "speakers": spk,
            "dialogue_ids": dia_ids, "utterance_ids": utt_ids,
            "length": torch.tensor(T, dtype=torch.long),
            "dialog_id": s.get("dialog_id", -1),
        }

def pad_sequence_2d(seqs: List[torch.Tensor], dim: int) -> torch.Tensor:
    Tmax = max(x.size(0) for x in seqs)
    B = len(seqs)
    out = torch.zeros(B, Tmax, dim, dtype=seqs[0].dtype)
    for i,x in enumerate(seqs):
        out[i, :x.size(0), :] = x
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
    dia_ids  = torch.full((B, Tmax), -1, dtype=torch.long)
    utt_ids  = torch.full((B, Tmax), -1, dtype=torch.long)

    for i,b in enumerate(batch):
        T = b["length"].item()
        labels[i, :T]   = b["labels"]
        speakers[i, :T] = b["speakers"]
        dia_ids[i, :T]  = b["dialogue_ids"]
        utt_ids[i, :T]  = b["utterance_ids"]

    return {
        "text": text, "audio": audio, "vision": vision,
        "labels": labels, "speakers": speakers, "length": lengths,
        "dialogue_ids": dia_ids, "utterance_ids": utt_ids
    }

# ---------------------------
# Inferencia
# ---------------------------

@torch.no_grad()
def predict(model: DM_CER, loader: DataLoader, device: str) -> List[Tuple[int,int,int]]:
    """
    Devuelve lista de (Dialogue_ID, Utterance_ID, pred_class_id)
    """
    model.eval()
    preds: List[Tuple[int,int,int]] = []
    for batch in tqdm(loader, desc="infer", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outs = model(batch, missing_modality_p=0.0)
        logits = outs["logits"]           # (B,T,C)
        labels = batch["labels"]          # (B,T) (pueden ser -100 en test)
        dia    = batch["dialogue_ids"]    # (B,T)
        utt    = batch["utterance_ids"]   # (B,T)

        mask = (dia >= 0) & (utt >= 0) & (labels != -100)  # si test no trae labels, podrías quitar labels != -100
        # Si tu test NO tiene labels, usa:
        # mask = (dia >= 0) & (utt >= 0)

        # Aplanar y recoger
        B,T,_ = logits.shape
        p = logits.argmax(dim=-1)  # (B,T)
        for i in range(B):
            for t in range(T):
                if mask[i,t].item():
                    preds.append((
                        int(dia[i,t].item()),
                        int(utt[i,t].item()),
                        int(p[i,t].item())
                    ))
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Carpeta con test.jsonl (y opcional dmcer_config_override.json)")
    ap.add_argument("--ckpt", required=True, help="Ruta a best.pt")
    ap.add_argument("--out_csv", required=True, help="CSV de salida con predicciones")
    ap.add_argument("--csv_in", default="", help="CSV original a enriquecer (se añadirá columna 'prediction')")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Config
    cfg = DMConfig()
    override_path = os.path.join(args.data_root, "dmcer_config_override.json")
    if os.path.isfile(override_path):
        with open(override_path, "r") as f:
            o = json.load(f)
        for k,v in o.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    test_jsonl = os.path.join(args.data_root, "test.jsonl")
    assert os.path.isfile(test_jsonl), f"No se encontró {test_jsonl}"

    # Dataset/Loader
    ds_te = MELDJSONLWithIDs(test_jsonl, cfg.d_text, cfg.d_audio, cfg.d_vision)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)

    # Modelo
    device = args.device
    model = DM_CER(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    # Inferencia
    preds = predict(model, dl_te, device)  # lista de (Dialogue_ID, Utterance_ID, pred_id)

    # Construir DataFrame de salida
    df_pred = pd.DataFrame(preds, columns=["Dialogue_ID","Utterance_ID","prediction"])
    df_pred["prediction"] = df_pred["prediction"].map(lambda x: EMO_ID2NAME.get(int(x), "neutral"))

    # Opción 1: si hay CSV de entrada, lo enriquecemos
    if args.csv_in:
        assert os.path.isfile(args.csv_in), f"No existe --csv_in {args.csv_in}"
        df_in = pd.read_csv(args.csv_in)
        # claves para merge
        if "Dialogue_ID" not in df_in.columns or "Utterance_ID" not in df_in.columns:
            raise ValueError("csv_in debe contener columnas Dialogue_ID y Utterance_ID")
        df_out = df_in.merge(df_pred, on=["Dialogue_ID","Utterance_ID"], how="left")
        df_out.to_csv(args.out_csv, index=False)
        print(f"Predicciones añadidas a {args.out_csv} (desde {args.csv_in})")
    else:
        # Opción 2: escribimos solo las columnas ID + prediction
        df_pred.to_csv(args.out_csv, index=False)
        print(f"CSV de predicciones escrito en {args.out_csv}")

if __name__ == "__main__":
    main()
