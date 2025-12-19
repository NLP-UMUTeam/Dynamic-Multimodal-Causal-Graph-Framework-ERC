
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse, tempfile, subprocess
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

import librosa
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

from facenet_pytorch import MTCNN
from PIL import Image
import cv2


# Map de emociones MELD -> IDs (ajusta si tu tarea usa otro mapeo)
EMOTION2ID = {
    "neutral": 0, "joy": 1, "sadness": 2, "anger": 3,
    "surprise": 4, "fear": 5, "disgust": 6
}

# ----------------------- Extractores -----------------------
def extract_text_emb(text, tokenizer, model, device):
    with torch.no_grad():
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        toks = {k:v.to(device) for k,v in toks.items()}
        out = model(**toks)
        cls = out.last_hidden_state[:,0,:].squeeze(0).cpu().numpy()  # (768,)
    return cls.tolist()

def ffmpeg_audio_to_wav(in_path, out_path, sr=16000):
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", str(sr), out_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def extract_audio_features(wav_path, sr=16000, n_mels=64):
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        return [0.0]* (2*n_mels)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(S + 1e-6)
    delta = librosa.feature.delta(logmel)
    # mean pooling over time
    m1 = logmel.mean(axis=1)   # (n_mels,)
    m2 = delta.mean(axis=1)    # (n_mels,)
    feat = np.concatenate([m1, m2], axis=0)  # (2*n_mels,) = 128
    return feat.astype(np.float32).tolist()

def sample_frame_indices(n_frames, k):
    if n_frames <= k:
        return list(range(n_frames))
    # uniform sampling
    return [int(round(i * (n_frames-1) / (k-1))) for i in range(k)]

def crop_faces_mtcnn(frame_rgb_uint8, mtcnn, min_prob=0.9, max_faces=5, margin_scale=1.2):
    """
    frame_rgb_uint8: np.uint8 (H,W,3) en RGB.
    Devuelve lista de (tensor_CHW_[0,1], area_px, score) de hasta 'max_faces' caras.
    """
    img_pil = Image.fromarray(frame_rgb_uint8)  # PIL RGB
    # MTCNN.detect devuelve boxes (x1,y1,x2,y2) y probs
    boxes, probs = mtcnn.detect(img_pil)
    crops = []
    if boxes is None or probs is None:
        return crops

    h, w, _ = frame_rgb_uint8.shape

    # Filtra por probabilidad mínima
    keep = [(box, float(p)) for box, p in zip(boxes, probs) if p is not None and p >= min_prob]
    if not keep:
        return crops

    # Ordena por (área * prob) descendente y limita a max_faces
    def area(b):
        x1, y1, x2, y2 = b
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    keep.sort(key=lambda bp: area(bp[0]) * bp[1], reverse=True)
    keep = keep[:max_faces]

    for box, score in keep:
        x1, y1, x2, y2 = box
        # Cuadro cuadrado + margen
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * margin_scale
        x1m = int(max(0, round(cx - side / 2)))
        y1m = int(max(0, round(cy - side / 2)))
        x2m = int(min(w, round(cx + side / 2)))
        y2m = int(min(h, round(cy + side / 2)))

        crop = frame_rgb_uint8[y1m:y2m, x1m:x2m]
        if crop.size == 0:
            continue
        # -> tensor CHW float [0,1]
        face_tensor = torch.from_numpy(crop).permute(2,0,1).float() / 255.0
        crops.append((face_tensor, float((x2m - x1m) * (y2m - y1m)), score))
       
    return crops

def extract_video_features(video_path, device, num_frames=8, mtcnn=None, image_processor=None, vmodel=None,
                           face_pool="area", min_prob=0.8, max_faces=10):
    try:
        import decord
        decord.bridge.set_bridge('torch')
        vr = decord.VideoReader(video_path)
        n = len(vr)
        if n == 0:
            return [0.0]*768   # dimensión de ViT
        idx = sample_frame_indices(n, num_frames)
        frames = vr.get_batch(idx).cpu().numpy()  # (k,H,W,3) uint8
    except Exception:
        try:
            from torchvision.io import read_video
            vframes, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")
            n = vframes.shape[0]
            if n == 0:
                return [0.0]*768
            idx = sample_frame_indices(n, num_frames)
            f = vframes[idx]  # (k,3,H,W)
            frames = f.permute(0,2,3,1).contiguous().cpu().numpy()
        except Exception:
            return [0.0]*768

    per_frame_feats = []
    with torch.no_grad():
        for i in range(frames.shape[0]):
            frame_rgb = frames[i]  # (H,W,3) uint8
            # detectar múltiples caras con MTCNN
            crops = crop_faces_mtcnn(frame_rgb, mtcnn, min_prob=min_prob, max_faces=max_faces)

            if len(crops) == 0:
                img = Image.fromarray(frame_rgb)
                inputs = image_processor(img, return_tensors="pt").to(device)
                out = vmodel(**inputs)
                feat = out.last_hidden_state[:,0,:].squeeze(0).cpu().numpy()  # [CLS] token
                per_frame_feats.append(feat)
                continue

            feats_faces = []
            weights_faces = []
            for face_tensor, area, score in crops:
                # tensor CHW float [0,1] -> PIL para HuggingFace
                face_img = Image.fromarray((face_tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                inputs = image_processor(face_img, return_tensors="pt").to(device)
                out = vmodel(**inputs)
                feat = out.last_hidden_state[:,0,:].squeeze(0).cpu().numpy()  # (768,)
                feats_faces.append(feat)
                weights_faces.append(area*score)

            feats_faces = np.stack(feats_faces, axis=0)
            if face_pool == "mean":
                f = feats_faces.mean(axis=0)
            elif face_pool == "area":
                w = np.array(weights_faces) / (np.sum(weights_faces)+1e-8)
                f = (feats_faces * w[:,None]).sum(axis=0)
            elif face_pool == "largest":
                f = feats_faces[np.argmax(weights_faces)]
            else:
                f = feats_faces.mean(axis=0)

            per_frame_feats.append(f)

    vid_feat = np.stack(per_frame_feats, axis=0).mean(axis=0).astype(np.float32)  # (768,)
    return vid_feat.tolist()
    

# ----------------------- Utilidades -----------------------
def rows_to_dialogs(df):
    dialogs = defaultdict(list)
    for _, r in df.iterrows():
        d = int(r["Dialogue_ID"])
        dialogs[d].append(r)
    for d in dialogs:
        dialogs[d] = sorted(dialogs[d], key=lambda r: int(r["Utterance_ID"]))
    return dialogs

def process_split(csv_path, video_base_path, out_jsonl, device, tokenizer, tmodel, mtcnn, image_processor, vmodel):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    df = pd.read_csv(csv_path)
    dialogs = rows_to_dialogs(df)

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for dialog_id, rows in tqdm(dialogs.items(), desc=f"Procesando {os.path.basename(csv_path)}"):
            turns = []
            speakers_seen = {}
            spk_counter = 0
            for _, r in enumerate(rows):
                utterance_id = int(r["Utterance_ID"])
                text = str(r["Utterance"])
                emotion = str(r["Emotion"]).lower()
                label = EMOTION2ID.get(emotion, 0)

                video_path = os.path.join(video_base_path, f"dia{dialog_id}_utt{utterance_id}.mp4")
                if not os.path.isfile(video_path):
                    # Si falta el video, salta el turno
                    continue

                # TEXT
                text_emb = extract_text_emb(text, tokenizer, tmodel, device)  # 768

                # AUDIO
                with tempfile.TemporaryDirectory() as td:
                    wav_path = os.path.join(td, "a.wav")
                    try:
                        ffmpeg_audio_to_wav(video_path, wav_path, sr=16000)
                        audio_feat = extract_audio_features(wav_path, sr=16000, n_mels=64)  # 128
                    except Exception:
                        audio_feat = [0.0]*128

                # VISION
                try:
                    vision_feat = extract_video_features(video_path, device, num_frames=8, mtcnn=mtcnn, image_processor=image_processor, vmodel=vmodel)  # 512
                except Exception:
                    vision_feat = [0.0]*768

                # SPEAKER
                spk_name = str(r.get("Speaker", "UNKNOWN"))
                if spk_name not in speakers_seen:
                    speakers_seen[spk_name] = spk_counter
                    spk_counter += 1
                speaker_idx = speakers_seen[spk_name]

                turns.append({
                    "speaker_idx": int(speaker_idx),
                    "text_emb": text_emb,
                    "audio": audio_feat,
                    "vision": vision_feat,
                    "label": int(label),
                    "dialogue_id": int(dialog_id),
                    "utterance_id": int(utterance_id)
                })

            if len(turns) == 0:
                continue

            sample = {
                "dialog_id": int(dialog_id),
                "speakers": list(speakers_seen.keys()),
                "turns": turns
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

# ----------------------- Main -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=False, help="Ruta CSV train_sent_emo.csv")
    parser.add_argument("--train_video_base", required=False, help="Base path de videos de train")
    parser.add_argument("--dev_csv", required=False, help="Ruta CSV dev_sent_emo.csv")
    parser.add_argument("--dev_video_base", required=False, help="Base path de videos de dev")
    parser.add_argument("--test_csv", required=False, help="Ruta CSV test_sent_emo.csv")
    parser.add_argument("--test_video_base", required=False, help="Base path de videos de test")
    parser.add_argument("--out_dir", required=True, help="Directorio de salida para JSONL")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Carga modelos compartidos (texto/visión) una sola vez
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tmodel = AutoModel.from_pretrained("bert-base-uncased").to(args.device).eval()
    
    vision_model_id = "mo-thecreator/vit-Facial-Expression-Recognition"
    image_processor = AutoImageProcessor.from_pretrained(vision_model_id)
    vmodel = AutoModel.from_pretrained(vision_model_id).to(args.device).eval()

    mtcnn = MTCNN(
        image_size=None,          # mantenemos tamaño original
        margin=0,                 # márgenes los aplicamos nosotros
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False,       # queremos recorte crudo; normalizamos nosotros
        device=args.device
    )
    
    # Procesa cada split si se proporcionó
    if args.train_csv and args.train_video_base:
        process_split(args.train_csv, args.train_video_base,
                      os.path.join(args.out_dir, "train.jsonl"),
                      args.device, tokenizer, tmodel, mtcnn, image_processor, vmodel)
    if args.dev_csv and args.dev_video_base:
        process_split(args.dev_csv, args.dev_video_base,
                      os.path.join(args.out_dir, "dev.jsonl"),
                      args.device, tokenizer, tmodel, mtcnn, image_processor, vmodel)
    if args.test_csv and args.test_video_base:
        process_split(args.test_csv, args.test_video_base,
                      os.path.join(args.out_dir, "test.jsonl"),
                      args.device, tokenizer, tmodel, mtcnn, image_processor, vmodel)

    # Config override con dimensiones de features
    with open(os.path.join(args.out_dir, "dmcer_config_override.json"), "w") as f:
        json.dump({"d_audio":128, "d_vision":768, "d_text":768}, f)

    print("Hecho. Archivos JSONL y dmcer_config_override.json generados en:", args.out_dir)

if __name__ == "__main__":
    main()
