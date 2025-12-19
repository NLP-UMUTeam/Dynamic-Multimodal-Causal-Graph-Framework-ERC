#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse, tempfile, subprocess, re
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
torch.backends.cudnn.benchmark = True

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from facenet_pytorch import MTCNN
from PIL import Image

# ----------------------- Mapeo de emociones -----------------------
# Si 'label' es string, se mapea; si ya es entero, se usa tal cual.
EMOTION2ID = {
    "neu": 0, "fru": 1, "ang": 2, "sad": 3,
    "hap": 4, "exc": 5,
}

# ----------------------- Helpers -----------------------
def dialog_key_from_uttr_id(uttr_id: str) -> str:
    m = re.match(r"^(Ses\d{2}[FM]_impro\d{2})_[FM]\d{3}$", uttr_id)
    if m: 
        return m.group(1)
    parts = uttr_id.split("_")
    return "_".join(parts[:-1]) if len(parts) >= 3 else uttr_id

def speaker_from_uttr_id(uttr_id: str) -> str:
    m = re.search(r"_([FM])\d{3}$", uttr_id)
    return m.group(1) if m else "UNK"

def rows_to_dialogs_iemocap(df):
    groups = defaultdict(list)
    for _, row in df.iterrows():
        uid = str(row["utt_id"])
        groups[dialog_key_from_uttr_id(uid)].append(row)

    def turn_index(uid: str) -> int:
        m = re.search(r"_[FM](\d{3})$", uid)
        return int(m.group(1)) if m else 0

    for dkey in groups:
        groups[dkey] = sorted(groups[dkey], key=lambda r: turn_index(str(r["utt_id"])))
    return groups

# ----------------------- Texto -----------------------
@torch.no_grad()
def extract_text_emb(text, tokenizer, model, device):
    text = "" if text is None or str(text).lower() == "nan" else str(text)
    if len(text.strip()) == 0:
        return [0.0] * 768
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    out = model(**toks)
    cls = out.last_hidden_state[:, 0, :].squeeze(0).float().cpu().numpy()
    return cls.tolist()

# ----------------------- Audio -----------------------
def ffmpeg_audio_segment_to_wav(in_path, out_path, start, end, sr=16000, hwaccel=False):
    dur = max(0.0, float(end) - float(start))
    if dur <= 0.0:
        raise ValueError("Segmento de audio con duración <= 0.")
    cmd = ["ffmpeg", "-y"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-ss", f"{start:.6f}",
        "-i", in_path,
        "-t", f"{dur:.6f}",
        "-ac", "1", "-ar", str(sr),
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def extract_audio_features(wav_path, sr=16000, n_mels=64):
    import librosa
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        return [0.0] * (2 * n_mels)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(S + 1e-6)
    delta = librosa.feature.delta(logmel)
    m1 = logmel.mean(axis=1); m2 = delta.mean(axis=1)
    feat = np.concatenate([m1, m2], axis=0).astype(np.float32)
    return feat.tolist()

# ----------------------- Vídeo + Caras -----------------------
def sample_frame_indices_between(start_f, end_f, k):
    total = max(0, end_f - start_f + 1)
    if total <= 0:
        return []
    if total <= k:
        return list(range(start_f, end_f + 1))
    return [int(round(start_f + i * (total - 1) / (k - 1))) for i in range(k)]

def crop_faces_mtcnn(frame_rgb_uint8, mtcnn, min_prob=0.9, max_faces=5, margin_scale=1.2):
    img_pil = Image.fromarray(frame_rgb_uint8)
    boxes, probs = mtcnn.detect(img_pil)
    crops = []
    if boxes is None or probs is None:
        return crops
    h, w, _ = frame_rgb_uint8.shape
    keep = [(box, float(p)) for box, p in zip(boxes, probs) if p is not None and p >= min_prob]
    if not keep:
        return crops
    def area(b):
        x1, y1, x2, y2 = b
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    keep.sort(key=lambda bp: area(bp[0]) * bp[1], reverse=True)
    keep = keep[:max_faces]
    for box, score in keep:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * margin_scale
        x1m = int(max(0, round(cx - side / 2))); y1m = int(max(0, round(cy - side / 2)))
        x2m = int(min(w, round(cx + side / 2))); y2m = int(min(h, round(cy + side / 2)))
        crop = frame_rgb_uint8[y1m:y2m, x1m:x2m]
        if crop.size == 0:
            continue
        crops.append((crop, float((x2m - x1m) * (y2m - y1m)), score))
    return crops

@torch.no_grad()
def extract_video_features_segment(
    video_path, device, start_sec, end_sec, num_frames=8, mtcnn=None,
    image_processor=None, vmodel=None, face_pool="area", min_prob=0.8,
    max_faces=10, batch_size=32, use_decord_gpu=True
):
    # 1) Leer frames del segmento
    frames_np = None
    try:
        import decord
        if use_decord_gpu and torch.cuda.is_available():
            ctx = decord.gpu(0)
        else:
            ctx = decord.cpu(0)
        vr = decord.VideoReader(video_path, ctx=ctx)
        n = len(vr)
        if n == 0:
            return [0.0] * 768
        fps = vr.get_avg_fps()
        start_f = max(0, int(round(start_sec * fps)))
        end_f = min(n - 1, int(round(end_sec * fps)))
        if end_f < start_f:
            end_f = start_f
        idx = sample_frame_indices_between(start_f, end_f, num_frames)
        if len(idx) == 0:
            return [0.0] * 768
        batch = vr.get_batch(idx)  # (k,H,W,3) on GPU if ctx=gpu(0)
        frames_np = batch.asnumpy() if hasattr(batch, "asnumpy") else batch.cpu().numpy()
        frames_np = frames_np.astype(np.uint8)
    except Exception:
        try:
            from torchvision.io import read_video
            vframes, _, _ = read_video(video_path, start_pts=float(start_sec), end_pts=float(end_sec),
                                       pts_unit="sec", output_format="TCHW")
            n = vframes.shape[0]
            if n == 0:
                return [0.0] * 768
            idx_local = sample_frame_indices_between(0, n - 1, num_frames)
            f = vframes[idx_local]  # (k,3,H,W)
            frames_np = f.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)
        except Exception:
            return [0.0] * 768

    # 2) Detectar caras y preparar batch de imágenes a ViT
    per_frame_feats = []
    face_imgs, face_wts, frame_has_faces = [], [], []

    for i in range(frames_np.shape[0]):
        frame_rgb = frames_np[i]
        crops = crop_faces_mtcnn(frame_rgb, mtcnn, min_prob=min_prob, max_faces=max_faces)
        if len(crops) == 0:
            frame_has_faces.append(False)
            face_imgs.append(Image.fromarray(frame_rgb))
            face_wts.append([1.0])  # peso dummy para marco sin caras
        else:
            frame_has_faces.append(True)
            wts = []
            imgs = []
            for crop_rgb, area, score in crops:
                imgs.append(Image.fromarray(crop_rgb))
                wts.append(area * score)
            face_imgs.append(imgs)
            face_wts.append(wts)

    # 3) Pasa por ViT en batch
    def encode_images_batch(pil_list):
        feats = []
        for i in range(0, len(pil_list), batch_size):
            batch_pil = pil_list[i:i+batch_size]
            inputs = image_processor(batch_pil, return_tensors="pt").to(device)
            out = vmodel(**inputs)
            f = out.last_hidden_state[:, 0, :].detach()  # (B,768)
            feats.append(f)
        return torch.cat(feats, dim=0) if len(feats) > 0 else torch.empty(0, 768, device=device)

    # Aplanar para batch único
    flat_imgs = []
    idx_ranges = []
    for imgs in face_imgs:
        if isinstance(imgs, list):
            idx_ranges.append((len(flat_imgs), len(flat_imgs) + len(imgs)))
            flat_imgs.extend(imgs)
        else:
            idx_ranges.append((len(flat_imgs), len(flat_imgs) + 1))
            flat_imgs.append(imgs)

    if len(flat_imgs) == 0:
        return [0.0] * 768

    all_feats = encode_images_batch(flat_imgs)  # (N,768) en device
    all_feats = all_feats.float()

    # 4) Pooling por frame
    ptr = 0
    for i, (start_i, end_i) in enumerate(idx_ranges):
        f_slice = all_feats[start_i:end_i]  # (n_i,768)
        if frame_has_faces[i]:
            w = torch.tensor(face_wts[i], device=all_feats.device, dtype=torch.float32)
            w = w / (w.sum() + 1e-8)
            fvec = (f_slice * w.unsqueeze(1)).sum(dim=0)
        else:
            fvec = f_slice[0]
        per_frame_feats.append(fvec)

    vid_feat = torch.stack(per_frame_feats, dim=0).mean(dim=0).detach().cpu().numpy().astype(np.float32)
    return vid_feat.tolist()

# ----------------------- Pipeline principal -----------------------
def process_tsv(tsv_path, out_jsonl, device, tokenizer, tmodel, mtcnn, image_processor, vmodel,
                num_frames=8, sr=16000, hwaccel_ffmpeg=False, vit_batch=32, use_decord_gpu=True):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    df = pd.read_csv(tsv_path, sep="\t")
    required = ["utt_id", "start_time", "end_time", "video_path", "label"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Falta la columna requerida: {c}")
    if "transcript" not in df.columns:
        df["transcript"] = ""

    dialogs = rows_to_dialogs_iemocap(df)
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for dialog_key, rows in tqdm(dialogs.items(), desc=f"Procesando {os.path.basename(tsv_path)}"):
            turns, speakers_seen, spk_counter = [], {}, 0
            for _, r in enumerate(rows):
                uttr_id = str(r["utt_id"])
                start = float(r["start_time"]); end = float(r["end_time"])
                video_path = str(r["video_path"])
                transcript = str(r["transcript"]) if pd.notna(r["transcript"]) else ""

                if not os.path.isfile(video_path):
                    continue

                # label puede ser int o str
                lbl = r["label"]
                if pd.api.types.is_number(lbl):
                    label = int(lbl)
                else:
                    label = EMOTION2ID.get(str(lbl).lower(), EMOTION2ID["neu"])

                # SPEAKER (F/M)
                spk_tag = speaker_from_uttr_id(uttr_id)
                if spk_tag not in speakers_seen:
                    speakers_seen[spk_tag] = spk_counter; spk_counter += 1
                speaker_idx = speakers_seen[spk_tag]

                # TEXT
                text_emb = extract_text_emb(transcript, tokenizer, tmodel, device)  # 768

                # AUDIO
                with tempfile.TemporaryDirectory() as td:
                    wav_path = os.path.join(td, "a.wav")
                    try:
                        ffmpeg_audio_segment_to_wav(video_path, wav_path, start, end, sr=sr, hwaccel=hwaccel_ffmpeg)
                        audio_feat = extract_audio_features(wav_path, sr=sr, n_mels=64)  # 128
                    except Exception:
                        audio_feat = [0.0] * 128

                # VISION
                try:
                    vision_feat = extract_video_features_segment(
                        video_path, device, start, end, num_frames=num_frames, mtcnn=mtcnn,
                        image_processor=image_processor, vmodel=vmodel, face_pool="area",
                        min_prob=0.8, max_faces=10, batch_size=vit_batch, use_decord_gpu=use_decord_gpu
                    )  # 768
                except Exception:
                    vision_feat = [0.0] * 768

                turns.append({
                    "speaker_idx": int(speaker_idx),
                    "text_emb": text_emb,
                    "audio": audio_feat,
                    "vision": vision_feat,
                    "label": int(label),
                    "dialogue_key": dialog_key,
                    "utterance_id": uttr_id,
                    "start": float(start),
                    "end": float(end),
                    "video_path": video_path
                })

            if len(turns) == 0:
                continue

            sample = {
                "dialog_id": dialog_key,             # p.ej. 'Ses01F_impro01'
                "speakers": list(speakers_seen.keys()),  # típicamente ['F','M']
                "turns": turns
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

# ----------------------- Main -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True, help="Ruta al TSV de IEMOCAP (uttr_id,start,end,video_path,label,transcript)")
    p.add_argument("--out_jsonl", required=True, help="Salida JSONL (un único split)")
    p.add_argument("--out_dir", required=True, help="Directorio para dmcer_config_override.json")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--sr", type=int, default=16000)

    # Aceleración
    p.add_argument("--ffmpeg-hwaccel", action="store_true", help="Habilita -hwaccel cuda en FFmpeg")
    p.add_argument("--vit-batch", type=int, default=32, help="Batch size para ViT")
    p.add_argument("--no-decord-gpu", action="store_true", help="Fuerza decodificación CPU en decord")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Modelos compartidos
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tmodel = AutoModel.from_pretrained("bert-base-uncased").to(args.device).eval()

    vision_model_id = "mo-thecreator/vit-Facial-Expression-Recognition"
    image_processor = AutoImageProcessor.from_pretrained(vision_model_id)
    vmodel = AutoModel.from_pretrained(vision_model_id).to(args.device).eval()

    mtcnn = MTCNN(
        image_size=None, margin=0, min_face_size=40,
        thresholds=[0.6, 0.7, 0.7], post_process=False,
        device=args.device
    )

    process_tsv(
        args.tsv, args.out_jsonl, args.device,
        tokenizer, tmodel, mtcnn, image_processor, vmodel,
        num_frames=args.num_frames, sr=args.sr,
        hwaccel_ffmpeg=args.ffmpeg_hwaccel,
        vit_batch=args.vit_batch,
        use_decord_gpu=not args.no_decord_gpu
    )

    # Config override con dimensiones de features
    with open(os.path.join(args.out_dir, "dmcer_config_override.json"), "w") as f:
        json.dump({"d_audio": 128, "d_vision": 768, "d_text": 768}, f)

    print("Hecho. JSONL y dmcer_config_override.json generados en:", args.out_dir)

if __name__ == "__main__":
    main()
