#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, argparse
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd

from model import DM_CER, DMConfig

# ---------------------------
# Dataset con IDs string (IEMOCAP)
# ---------------------------

class IEMOCAPJSONLWithIDs(Dataset):
    """
    Cada línea del JSONL (prepro IEMOCAP):
    {
      "dialog_id": "Ses01F_impro01",
      "speakers": ["F","M"],
      "turns": [
        {
          "speaker_idx": int,
          "text_emb": [...], "audio": [...], "vision": [...],
          "label": int (opcional en test),
          "utterance_id": "Ses01F_impro01_F000",  # string
          "dialogue_key": "Ses01F_impro01",       # string
          ...
        }, ...
      ]
    }
    """
    def __init__(self, jsonl_path: str, d_text: int, d_audio: int, d_vision: int, expect_labels: bool = False):
        assert os.path.isfile(jsonl_path), f"No existe {jsonl_path}"
        self.path = jsonl_path
        self.d_text, self.d_audio, self.d_vision = d_text, d_audio, d_vision
        self.expect_labels = expect_labels
        self.samples: List[Dict[str,Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        turns = s["turns"]
        T = len(turns)

        text   = torch.tensor([t["text_emb"] for t in turns], dtype=torch.float32)    # (T,d_text)
        audio  = torch.tensor([t["audio"]     for t in turns], dtype=torch.float32)   # (T,d_audio)
        vision = torch.tensor([t["vision"]    for t in turns], dtype=torch.float32)   # (T,d_vision)

        if self.expect_labels:
            labels = torch.tensor([t.get("label", -100) for t in turns], dtype=torch.long)
        else:
            # si test no tiene labels, usa todo -100
            labels = torch.full((T,), -100, dtype=torch.long)

        speakers = torch.tensor([t["speaker_idx"] for t in turns], dtype=torch.long)

        # IDs string que necesitamos para exportar/merge
        utt_keys = [str(t.get("utterance_id", "")) for t in turns]               # p.ej. "Ses01F_impro01_F000"
        dia_keys = [str(t.get("dialogue_key", s.get("dialog_id", ""))) for t in turns]  # "Ses01F_impro01"

        return {
            "text": text, "audio": audio, "vision": vision,
            "labels": labels, "speakers": speakers,
            "utt_keys": utt_keys, "dia_keys": dia_keys,
            "length": torch.tensor(T, dtype=torch.long),
            "dialog_id": s.get("dialog_id", ""),
        }

def pad_sequence_2d(seqs: List[torch.Tensor], dim: int) -> torch.Tensor:
    Tmax = max(x.size(0) for x in seqs)
    B = len(seqs)
    out = torch.zeros(B, Tmax, dim, dtype=seqs[0].dtype)
    for i,x in enumerate(seqs):
        out[i, :x.size(0), :] = x
    return out

def collate_batch(batch: List[Dict[str,Any]]) -> Dict[str, Any]:
    B = len(batch)
    lengths = torch.stack([b["length"] for b in batch])
    Tmax = int(lengths.max().item())

    text   = pad_sequence_2d([b["text"] for b in batch],   batch[0]["text"].size(1))
    audio  = pad_sequence_2d([b["audio"] for b in batch],  batch[0]["audio"].size(1))
    vision = pad_sequence_2d([b["vision"] for b in batch], batch[0]["vision"].size(1))

    labels   = torch.full((B, Tmax), -100, dtype=torch.long)
    speakers = torch.zeros(B, Tmax, dtype=torch.long)

    # IDs string: mantenemos listas de longitud variable por elemento del batch
    utt_keys_batch: List[List[str]] = []
    dia_keys_batch: List[List[str]] = []

    for i,b in enumerate(batch):
        T = b["length"].item()
        labels[i, :T]   = b["labels"]
        speakers[i, :T] = b["speakers"]
        utt_keys_batch.append(b["utt_keys"])  # len T (strings)
        dia_keys_batch.append(b["dia_keys"])

    return {
        "text": text, "audio": audio, "vision": vision,
        "labels": labels, "speakers": speakers, "length": lengths,
        "utt_keys": utt_keys_batch, "dia_keys": dia_keys_batch
    }

# ---------------------------
# Inferencia
# ---------------------------

@torch.no_grad()
def predict(model: DM_CER, loader: DataLoader, device: str) -> List[Tuple[str, str, int]]:
    """
    Devuelve lista de (dialogue_key, utt_id, pred_class_id) todos como strings salvo la clase.
    """
    model.eval()
    preds: List[Tuple[str, str, int]] = []
    for batch in tqdm(loader, desc="infer", leave=False):
        # tensores a device; listas (IDs) se quedan en CPU
        tens_keys = ["text","audio","vision","labels","speakers","length"]
        batch_t = {k: v.to(device) for k, v in batch.items() if k in tens_keys}
        outs = model(batch_t, missing_modality_p=0.0)
        logits = outs["logits"]           # (B,T,C)

        B, T, _ = logits.shape
        p = logits.argmax(dim=-1).cpu().numpy()  # (B,T) en CPU para indexar fácil

        # Recuperar listas de keys string
        utt_keys_batch: List[List[str]] = batch["utt_keys"]
        dia_keys_batch: List[List[str]] = batch["dia_keys"]
        lengths = batch_t["length"].cpu().numpy().tolist()

        for i in range(B):
            Ti = int(lengths[i])
            for t in range(Ti):
                preds.append((
                    str(dia_keys_batch[i][t]),
                    str(utt_keys_batch[i][t]),
                    int(p[i, t])
                ))
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Carpeta con iemocap_test.jsonl, emotion_map.json y (opcional) dmcer_config_override.json")
    ap.add_argument("--ckpt", required=True, help="Ruta a best.pt")
    ap.add_argument("--out_tsv", required=True, help="TSV de salida con predicciones")
    ap.add_argument("--tsv_in", default="", help="TSV original a enriquecer (se añadirá columna 'prediction' mapeada a nombre)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Config (dimensiones)
    cfg = DMConfig()
    override_path = os.path.join(args.data_root, "dmcer_config_override.json")
    if os.path.isfile(override_path):
        with open(override_path, "r") as f:
            o = json.load(f)
        for k,v in o.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # Carga mapeo de emociones detectado en prepro
    id2label = {
        0:"neu", 1:"fru", 2:"ang", 3:"sad",
        4:"hap", 5:"exc",
    }

    # Paths
    test_jsonl = os.path.join(args.data_root, "iemocap_test.jsonl")
    assert os.path.isfile(test_jsonl), f"No se encontró {test_jsonl}"

    # Dataset/Loader
    ds_te = IEMOCAPJSONLWithIDs(test_jsonl, cfg.d_text, cfg.d_audio, cfg.d_vision, expect_labels=False)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)

    # Modelo
    device = args.device
    model = DM_CER(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    # Inferencia
    preds = predict(model, dl_te, device)  # lista de (dialogue_key, utt_id, pred_id)
    # Preds -> DataFrame
    df_pred = pd.DataFrame(preds, columns=["dialogue_key","utt_id","prediction_id"])
    df_pred["prediction"] = df_pred["prediction_id"].map(lambda x: id2label.get(int(x), "neutral"))
    print(df_pred)
    
    if args.tsv_in:
        assert os.path.isfile(args.tsv_in), f"No existe --tsv_in {args.tsv_in}"
        df_in = pd.read_csv(args.tsv_in, sep="\t")

        # Si falta dialogue_key, lo derivamos de utt_id (p. ej. "Ses01F_impro01_F000" -> "Ses01F_impro01")
        if "dialogue_key" not in df_in.columns:
            import re
            def to_dialogue_key(u):
                if not isinstance(u, str): return ""
                m = re.match(r"^(Ses\d{2}[FM]_impro\d{2})_[FM]\d{3}$", u)
                return m.group(1) if m else ""
            df_in["dialogue_key"] = df_in["utt_id"].astype(str).map(to_dialogue_key)

        # Comprobaciones de claves
        if "utt_id" not in df_in.columns:
            raise ValueError("--tsv_in debe contener columna 'utt_id'")

        # Merge por (dialogue_key, utt_id)
        df_out = df_in.merge(
            df_pred[["dialogue_key","utt_id","prediction"]],
            on=["utt_id"],
            how="left"
        )
        df_out.to_csv(args.out_tsv, index=False, sep="\t")
        print(f"Predicciones añadidas a {args.out_tsv} (desde {args.tsv_in})")

    else:
        # Solo claves + predicción
        df_pred[["dialogue_key","utt_id","prediction"]].to_csv(args.out_tsv, index=False, sep="\t")
        print(f"TSV de predicciones escrito en {args.out_tsv}")
    

if __name__ == "__main__":
    main()
