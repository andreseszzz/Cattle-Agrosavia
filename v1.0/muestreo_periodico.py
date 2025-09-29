#!/usr/bin/env python3
"""
select_frames_periodic.py


Enfoque 1 - muestreo periódico:
Extrae frames cada N segundos de cada .mp4 encontrado bajo input_dir,
pasa cada frame al modelo de segmentación y, si detecta >=1 vaca válida,
guarda imagen, máscara y JSON de anotación en output_dir manteniendo la estructura relativa.


Uso (ejemplo):
python muestreo_periodico.py \
 --input_dir "/home/pipe/Documentos/yolov11/videos" \
 --output_dir "/home/pipe/Documentos/yolov11/videos/results" \
 --interval 15 \
 --model_type custom \
 --model_path "/home/pipe/Documentos/yolov11/muestreo_periodico/sam_vit_h_4b8939.pth" \
 --sam_type vit_h \
 --yolo_model_path "/home/pipe/Documentos/yolov11/muestreo_periodico/best.pt" \
 --device cpu \
 --score_thresh 0.5 \
 --min_mask_area 500


Dependencias:
pip install opencv-python pillow numpy
# opcional si usas PyTorch / ONNX:
pip install torch onnxruntime
"""


import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import json
import datetime
import sys
import os
import torch


H264_EXTS = {".h264", ".264", ".bin"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}  # procesaremos mp4 principalmente


# ---------------- utilities ----------------
def ensure_dir(p: Path):
   p.mkdir(parents=True, exist_ok=True)


def save_mask_png(mask: np.ndarray, out_path: Path):
   # mask: 0/1 or boolean -> save 0/255 PNG
   img = Image.fromarray((mask.astype(np.uint8) * 255))
   img.save(str(out_path))


def mask_to_bbox(mask: np.ndarray):
   ys, xs = np.where(mask > 0)
   if ys.size == 0:
       return None
   xmin, xmax = int(xs.min()), int(xs.max())
   ymin, ymax = int(ys.min()), int(ys.max())
   w = xmax - xmin + 1
   h = ymax - ymin + 1
   return [xmin, ymin, w, h]


def is_valid_file(p: Path):
   try:
       return p.exists() and p.is_file() and p.stat().st_size > 0
   except Exception:
       return False


# ---------------- Model wrapper (placeholder) ----------------
class ModelWrapper:
   """
   En este wrapper puedes integrar tu modelo de segmentación.
   Soporta 3 modos:
     - 'pytorch' : intenta cargar un JIT/Traced model o modelo simple (adapta si usas state_dict)
     - 'onnx'    : carga ONNX con onnxruntime
     - 'custom'  : usa la función run_inference_custom(frame_bgr) definida más abajo
   IMPORTANTE: adapta _run_pytorch/_run_onnx a la API de salida de tu modelo si es necesario.
   """
   def __init__(self, model_type: str, model_path: str = "", device: str = "cpu"):
       self.type = model_type.lower()
       self.device = device
       self.model_path = model_path
       self.model = None
       if self.type == "pytorch":
           try:
               import torch
               self.torch = torch
           except Exception as e:
               raise ImportError("Para usar 'pytorch' instala torch. Error: " + str(e))
           self._load_pytorch(model_path)
       elif self.type == "onnx":
           try:
               import onnxruntime as ort
               self.ort = ort
           except Exception as e:
               raise ImportError("Para usar 'onnx' instala onnxruntime. Error: " + str(e))
           self._load_onnx(model_path)
       elif self.type == "custom":
           print("[ModelWrapper] modo custom: se usará run_inference_custom(frame_bgr).")
           # Guarda la ruta del modelo para la función custom
           global my_model_path
           my_model_path = model_path
       else:
           raise ValueError("model_type no soportado. Usa 'pytorch','onnx' o 'custom'.")


   def _load_pytorch(self, path):
       import torch
       if not path:
           raise ValueError("Para model_type='pytorch' debes pasar --model_path")
       print("Cargando PyTorch model desde:", path)
       # Intento cargar JIT o modelo completo.
       try:
           self.model = torch.jit.load(path, map_location=self.device)
           print("Modelo PyTorch JIT cargado.")
       except Exception:
           loaded = torch.load(path, map_location=self.device)
           if isinstance(loaded, dict):
               raise RuntimeError("Se cargó un state_dict. Debes instanciar la arquitectura y adaptar el loader.")
           else:
               self.model = loaded
       self.model.to(self.device)
       self.model.eval()


   def _load_onnx(self, path):
       import onnxruntime as ort
       if not path:
           raise ValueError("Para model_type='onnx' debes pasar --model_path")
       print("Cargando ONNX model desde:", path)
       self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
       print("ONNX session creada.")


   def run(self, frame_bgr: np.ndarray):
       """
       Entrada: frame BGR (H,W,3) uint8
       Salida esperada: mask (H,W) binaria np.uint8, class_name (str), score (float)
       ADAPTA según la salida de tu modelo.
       """
       if self.type == "pytorch":
           return self._run_pytorch(frame_bgr)
       elif self.type == "onnx":
           return self._run_onnx(frame_bgr)
       else:
           return run_inference_custom(frame_bgr)


   def _run_pytorch(self, frame_bgr):
       import torch
       # Ejemplo genérico: convertir a RGB y normalizar 0-1. ADAPTA a tu preprocessing.
       img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
       tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
       tensor = tensor.to(self.device)
       with torch.no_grad():
           out = self.model(tensor)
       # Aquí asumimos que `out` es una máscara de probabilidad (1,H,W) o (H,W)
       if isinstance(out, (list, tuple)):
           out0 = out[0]
       else:
           out0 = out
       if torch.is_tensor(out0):
           arr = out0.squeeze().cpu().numpy()
           # Si la salida tiene 2D pero con valores en 0..1 -> umbralizar
           mask = (arr > 0.5).astype(np.uint8)
           score = float(arr.max())
           return mask, "vaca", score
       else:
           raise RuntimeError("Salida de modelo PyTorch inesperada. Adapta _run_pytorch.")


   def _run_onnx(self, frame_bgr):
       # ADAPTA según tu ONNX input/output names/sizes
       img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
       inp = np.transpose(img.astype(np.float32)/255.0, (2,0,1))[None,...]
       input_name = self.session.get_inputs()[0].name
       outputs = self.session.run(None, {input_name: inp})
       out0 = outputs[0]
       arr = np.squeeze(out0)
       mask = (arr > 0.5).astype(np.uint8)
       score = float(arr.max())
       return mask, "vaca", score


# ---------------- Placeholder custom inference ----------------
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO  # pip install ultralytics


def run_inference_custom(frame_bgr: np.ndarray):
   global sam_predictor
   global my_model_path
   global sam_type
   global yolo_model
   global yolo_model_path
   global device


   # Detecta si hay GPU disponible
   if 'device' not in globals():
       device = "cuda" if torch.cuda.is_available() else "cpu"


   # Cargar SAM solo una vez
   if 'sam_predictor' not in globals():
       sam = sam_model_registry[sam_type](checkpoint=my_model_path)
       sam.to(device)
       sam_predictor = SamPredictor(sam)


   # Cargar YOLO solo una vez (y pasar device)
   if 'yolo_model' not in globals():
       yolo_model = YOLO(yolo_model_path)
       yolo_model.to(device)


   img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


   # 1. Detecta vacas con YOLO
   results = yolo_model(img)
   boxes = results[0].boxes
   names = results[0].names


   best_mask = None
   best_score = 0.0
   best_class = None
   for box in boxes:
       class_id = int(box.cls[0])
       class_name = names[class_id]
       if class_name.lower() not in ["cow", "vaca"]:
           continue  # Solo vacas


       x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
       input_box = np.array([[x1, y1, x2, y2]])
       sam_predictor.set_image(img)
       masks, scores, logits = sam_predictor.predict(
           box=input_box,
           multimask_output=False
       )
       mask = masks[0].astype(np.uint8)
       score = float(scores[0])
       if score > best_score:
           best_mask = mask
           best_score = score
           best_class = class_name


   if best_mask is not None:
       return best_mask, best_class, best_score


   return None, None, 0.0


# ---------------- Main pipeline ----------------
import time


def process_periodic(input_root: Path, output_root: Path, model_wrapper: ModelWrapper,
                    interval: float = 15.0, score_thresh: float = 0.5, min_mask_area: int = 500,
                    dry_run: bool = False):
   input_root = input_root.expanduser().resolve()
   output_root = output_root.expanduser().resolve()
   if not input_root.is_dir():
       raise SystemExit(f"Input no es carpeta válida: {input_root}")
   ensure_dir(output_root)


   mp4_files = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
   print(f"Encontrados {len(mp4_files)} archivos de video en {input_root}")


   saved_count = 0
   processed_frames = 0
   mask_areas = []
   mask_scores = []


   start_time = time.time()


   for video_path in mp4_files:
       try:
           rel_parent = video_path.parent.relative_to(input_root)
       except Exception:
           rel_parent = Path(".")
       video_stem = video_path.stem
       out_images_dir = output_root / "images" / rel_parent / video_stem
       out_masks_dir = output_root / "masks" / rel_parent / video_stem
       out_ann_dir = output_root / "annotations" / rel_parent / video_stem
       if not dry_run:
           ensure_dir(out_images_dir); ensure_dir(out_masks_dir); ensure_dir(out_ann_dir)


       cap = cv2.VideoCapture(str(video_path))
       if not cap.isOpened():
           print(f"[WARN] No se pudo abrir video: {video_path}")
           continue
       fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
       duration = (total_frames / fps) if fps else 0.0
       print(f"\nProcesando: {video_path}  | FPS={fps:.2f}  frames={total_frames}  dur={duration:.1f}s")


       t = 0.0
       idx = 0
       while t < duration:
           cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0)
           ret, frame = cap.read()
           if not ret:
               break
           processed_frames += 1
           print(f"  frame @ {t:.1f}s (idx {idx})", end=" ... ")
           try:
               mask, class_name, score = model_wrapper.run(frame)
           except Exception as e:
               print(f"ERROR inferencia: {e}")
               idx += 1
               t += interval
               continue


           if mask is None:
               print("no mask -> descartado")
               idx += 1
               t += interval
               continue


           mask = (mask > 0).astype(np.uint8)
           mask_area = int(mask.sum())
           print(f"score={score:.3f}, mask_area={mask_area}", end=" ")


           if score >= score_thresh and mask_area >= min_mask_area:
               timestamp_iso = datetime.datetime.utcnow().isoformat() + "Z"
               file_stem = f"{video_stem}_frame_{idx:06d}"
               img_name = f"{file_stem}.jpg"
               mask_name = f"{file_stem}_mask.png"
               json_name = f"{file_stem}.json"


               img_path = out_images_dir / img_name
               mask_path = out_masks_dir / mask_name
               json_path = out_ann_dir / json_name


               if dry_run:
                   print("-> DETECCIÓN (dry_run) - se guardaría:", img_path)
               else:
                   cv2.imwrite(str(img_path), frame)
                   save_mask_png(mask, mask_path)
                   bbox = mask_to_bbox(mask)
                   metadata = {
                       "image_id": file_stem,
                       "file_name": str(img_path.relative_to(output_root)),
                       "mask_path": str(mask_path.relative_to(output_root)),
                       "video_path": str(video_path.relative_to(input_root)),
                       "frame_index_in_video": idx,
                       "timestamp_sec_in_video": t,
                       "timestamp_saved_utc": timestamp_iso,
                       "class": class_name,
                       "score": float(score),
                       "bbox": bbox,
                       "mask_area_px": mask_area
                   }
                   with open(json_path, "w", encoding="utf-8") as jf:
                       json.dump(metadata, jf, indent=2, ensure_ascii=False)
                   print("-> guardado")
               saved_count += 1
               mask_areas.append(mask_area)
               mask_scores.append(score)
           else:
               print("-> descartado")


           idx += 1
           t += interval


       cap.release()


   end_time = time.time()
   elapsed = end_time - start_time


   print("\n===== Resumen =====")
   print(f"Frames procesados: {processed_frames}")
   print(f"Frames guardados (detecciones): {saved_count}")
   print(f"Tiempo total de ejecución: {elapsed:.2f} segundos")


   if saved_count > 0:
       avg_area = np.mean(mask_areas)
       avg_score = np.mean(mask_scores)
       perc_detected = 100.0 * saved_count / processed_frames
       print(f"Porcentaje de frames con detección: {perc_detected:.2f}%")
       print(f"Área promedio de las máscaras: {avg_area:.1f} px")
       print(f"Score promedio de las detecciones: {avg_score:.3f}")
   else:
       print("No hubo detecciones para calcular métricas.")




# ---------------- CLI ----------------
def parse_args():
   p = argparse.ArgumentParser(description="Selector de frames (Enfoque 1 - muestreo periódico)")
   p.add_argument("--input_dir", required=True, help="Directorio raíz con videos (subcarpetas por fecha).")
   p.add_argument("--output_dir", required=True, help="Directorio donde se guardarán images/masks/annotations.")
   p.add_argument("--interval", type=float, default=15.0, help="Segundos entre frames a extraer (ej. 15 o 20).")
   p.add_argument("--model_type", choices=["pytorch","onnx","custom"], default="custom", help="Tipo de modelo (usa 'custom' para placeholder).")
   p.add_argument("--model_path", default="/home/pipe/Documentos/yolov11/sam_vit_h_4b8939.pth", help="Ruta al modelo SAM (.pt).")
   p.add_argument("--sam_type", choices=["vit_b","vit_l","vit_h"], default="vit_h", help="Tipo de arquitectura SAM.")
   p.add_argument("--yolo_model_path", default="/home/pipe/Documentos/yolov11/best.pt", help="Ruta al modelo YOLO (.pt).")
   p.add_argument("--device", default="cpu", help="Device para PyTorch (ej. 'cpu' o 'cuda:0').")
   p.add_argument("--score_thresh", type=float, default=0.5, help="Umbral de confianza minimo para aceptar detección.")
   p.add_argument("--min_mask_area", type=int, default=500, help="Min area en px de la máscara para aceptar detección.")
   p.add_argument("--dry_run", action="store_true", help="No escribe archivos, solo muestra qué haría.")
   return p.parse_args()


if __name__ == "__main__":
   args = parse_args()
   input_dir = Path(args.input_dir)
   output_dir = Path(args.output_dir)


   # Validaciones simples
   if not input_dir.exists():
       print("Input_dir no existe:", input_dir); sys.exit(1)
   if args.model_type != "custom" and not args.model_path:
       print("model_path requerido para model_type", args.model_type); sys.exit(1)


   # Asigna variables globales para SAM y YOLO
   global my_model_path
   global sam_type
   global yolo_model_path
   my_model_path = args.model_path
   sam_type = args.sam_type
   yolo_model_path = args.yolo_model_path


   mw = ModelWrapper(args.model_type, model_path=args.model_path, device=args.device)
   process_periodic(input_dir, output_dir, mw,
                    interval=args.interval,
                    score_thresh=args.score_thresh,
                    min_mask_area=args.min_mask_area,
                    dry_run=args.dry_run)
