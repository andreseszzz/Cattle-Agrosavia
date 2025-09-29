#!/usr/bin/env python3
import cv2
import os
import numpy as np
import torch
import json
from datetime import datetime, timezone


# ==============================
# Parámetros ajustables
# ==============================
FRAME_INTERVAL = 15        # Intervalo en segundos (Enfoque 1)
MIN_AREA = 500             # Área mínima para considerar movimiento por absdiff
OUTPUT_DIR = "/home/pipe/Documentos/yolov11/muestreo_hibrido"


# --- Modelos ---
MODEL_YOLO_PATH = "/home/pipe/Documentos/yolov11/muestreo_hibrido/yolo11m.pt"
SAM_CHECKPOINT = "/home/pipe/Documentos/yolov11/muestreo_hibrido/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
YOLO_CONF = 0.30
TARGET_CLASSES = "cow"
USE_SAM_VERIFICATION = True
SAVE_MASKS = True


# --- Gradiente ---
USE_GRADIENT = True
GRADIENT_THRESH = 20
GRADIENT_MIN_AREA = 300
GRAD_SOBEL_KSIZE = 3


# ==============================
# Dispositivo dinámico
# ==============================
_device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Usando dispositivo: {_device_str}")


# ==============================
# Carga de modelos (YOLO + SAM)
# ==============================
_yolo_model = None
_sam_predictor = None


def _load_models():
   global _yolo_model, _sam_predictor
   if _yolo_model is None:
       from ultralytics import YOLO
       _yolo_model = YOLO(MODEL_YOLO_PATH)
       try:
           _yolo_model.to(_device_str)
       except Exception:
           pass
   if USE_SAM_VERIFICATION and _sam_predictor is None:
       from segment_anything import sam_model_registry, SamPredictor
       sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
       sam.to(device=_device_str)
       _sam_predictor = SamPredictor(sam)


# ==============================
# Detección y segmentación
# ==============================
def detectar_vaca(frame, frame_id=None, video_name=None, timestamp=None):
   _load_models()
   results = _yolo_model(frame, verbose=False, conf=YOLO_CONF, device=_device_str)


   detections = []
   rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   h, w = frame.shape[:2]


   for r in results:
       names = r.names
       if getattr(r, "boxes", None) is None:
           continue
       for i, box in enumerate(r.boxes):
           cls_idx = int(box.cls.item())
           label = names.get(cls_idx, str(cls_idx)).lower()
           if label not in TARGET_CLASSES:
               continue


           xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
           score = float(box.conf.item())
           mask_area = None
           mask_path = None


           if USE_SAM_VERIFICATION and _sam_predictor is not None:
               _sam_predictor.set_image(rgb)
               x1c, y1c, x2c, y2c = xyxy
               masks, scores, _ = _sam_predictor.predict(
                   point_coords=None,
                   point_labels=None,
                   box=np.array([x1c, y1c, x2c, y2c]),
                   multimask_output=True,
               )
               if masks is not None and len(masks) > 0:
                   best_idx = int(np.argmax(scores))
                   best_mask = masks[best_idx].astype(np.uint8)
                   mask_area = int(best_mask.sum())
                   if SAVE_MASKS and frame_id is not None:
                       mask_dir = os.path.join(OUTPUT_DIR, "masks")
                       os.makedirs(mask_dir, exist_ok=True)
                       mask_filename = f"{video_name}_frame_{frame_id:06d}_obj{i}.png"
                       mask_path = os.path.join("masks", video_name, mask_filename)
                       cv2.imwrite(os.path.join(OUTPUT_DIR, "masks", mask_filename), best_mask * 255)


           detections.append({
               "class": label,
               "score": score,
               "bbox": xyxy,
               "mask_area_px": mask_area,
               "mask_path": mask_path
           })


   return detections


# ==============================
# Helpers gradiente / absdiff
# ==============================
def compute_gray_blur(frame, ksize=(21,21)):
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   return cv2.GaussianBlur(gray, ksize, 0)


def absdiff_motion(prev_gray, gray, thresh_val=25, min_area=MIN_AREA):
   diff = cv2.absdiff(prev_gray, gray)
   _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
   thresh = cv2.dilate(thresh, None, iterations=2)
   contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   return any(cv2.contourArea(c) > min_area for c in contours), thresh


def grad_magnitude(frame_gray, ksize=GRAD_SOBEL_KSIZE):
   gx = cv2.Sobel(frame_gray, cv2.CV_32F, 1, 0, ksize=ksize)
   gy = cv2.Sobel(frame_gray, cv2.CV_32F, 0, 1, ksize=ksize)
   mag = cv2.magnitude(gx, gy)
   mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
   return mag


def grad_diff_motion(prev_grad, cur_grad, grad_thresh=GRADIENT_THRESH, min_area=GRADIENT_MIN_AREA):
   diff = cv2.absdiff(prev_grad, cur_grad)
   _, thresh = cv2.threshold(diff, grad_thresh, 255, cv2.THRESH_BINARY)
   thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
   contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   return any(cv2.contourArea(c) > min_area for c in contours), thresh


# ==============================
# Procesar video con enfoque híbrido
# ==============================
def procesar_video(video_path):
   video_name = os.path.splitext(os.path.basename(video_path))[0]
   os.makedirs(os.path.join(OUTPUT_DIR, "frames"), exist_ok=True)
   os.makedirs(os.path.join(OUTPUT_DIR, "annotations"), exist_ok=True)
   if SAVE_MASKS:
       os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)


   cap = cv2.VideoCapture(video_path)
   fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
   fps = int(round(fps))
   frame_count = 0
   saved_count = 0


   ret, prev_frame = cap.read()
   if not ret:
       print("Error: no se pudo leer el primer frame.")
       return


   prev_gray = compute_gray_blur(prev_frame)
   prev_grad = grad_magnitude(prev_gray) if USE_GRADIENT else None


   while True:
       ret, frame = cap.read()
       if not ret:
           break


       frame_count += 1
       timestamp = frame_count / fps


       is_interval = (frame_count % (FRAME_INTERVAL * fps)) == 0
       gray = compute_gray_blur(frame)


       abs_motion_flag, _ = absdiff_motion(prev_gray, gray, thresh_val=25, min_area=MIN_AREA)


       grad_motion_flag = False
       if USE_GRADIENT:
           cur_grad = grad_magnitude(gray)
           grad_motion_flag, _ = grad_diff_motion(prev_grad, cur_grad, grad_thresh=GRADIENT_THRESH, min_area=GRADIENT_MIN_AREA)
       else:
           cur_grad = None


       prev_gray = gray
       if USE_GRADIENT:
           prev_grad = cur_grad


       is_motion = abs_motion_flag or grad_motion_flag


       if is_interval or is_motion:
           detections = detectar_vaca(frame, frame_id=frame_count, video_name=video_name, timestamp=timestamp)
           if detections:
               frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
               frame_path = os.path.join("frames", frame_filename)
               cv2.imwrite(os.path.join(OUTPUT_DIR, "frames", frame_filename), frame)
               saved_count += 1


               # anotación JSON
               ann = {
                   "image_id": f"{video_name}_frame_{frame_count:06d}",
                   "file_name": frame_path,
                   "video_path": video_name + ".mp4",
                   "frame_index_in_video": frame_count,
                   "timestamp_sec_in_video": timestamp,
                   "timestamp_saved_utc": datetime.now(timezone.utc).isoformat(),
                   "detections": detections
               }
               ann_path = os.path.join(OUTPUT_DIR, "annotations", f"{video_name}_frame_{frame_count:06d}.json")
               with open(ann_path, "w") as f:
                   json.dump(ann, f, indent=2)


               print(f"[✔] Guardado {frame_filename} con anotación JSON ({len(detections)} detecciones)")
           else:
               print(f"[✖] Frame {frame_count} descartado (no cow)")


   cap.release()
   print(f"✅ Proceso finalizado. Frames guardados: {saved_count}")


# ==============================
# Ejecución
# ==============================
if __name__ == "__main__":
   video_path = "/home/pipe/Documentos/yolov11/muestreo_hibrido/clip3_to_Mold.mp4"
   procesar_video(video_path)
