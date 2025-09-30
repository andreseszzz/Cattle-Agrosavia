import cv2
import os
import numpy as np
import json
from ultralytics import YOLO, SAM
from PIL import Image
import torch
import time
from datetime import datetime, timezone


# =======================
# CONFIGURACIONES
# =======================
VIDEO_PATH = r"/home/pipe/Documentos/Proyecto_Ganado/clip3_to_Mold.mp4"
OUTPUT_DIR = r"/home/pipe/Documentos/Proyecto_Ganado/v1.1/output_mog2_ROI"


SAM_PATH = r"/home/pipe/Documentos/Proyecto_Ganado/sam2.1_l.pt"
YOLO_PATH = r"/home/pipe/Documentos/Proyecto_Ganado/yolo12l.pt"


YOLO_CONF = 0.25
YOLO_IOU = 0.45
TARGET_CLASSES = ["cow"]


# Detección / muestreo
FRAME_SKIP = 15
MIN_AREA = 4000
MOG_HISTORY = 1000
MOG_VAR_THRESHOLD = 50
MOG_DETECT_SHADOWS = True


# Deduplicación
DUPLICATE_SAVE_DIFF = 5.0
DUPLICATE_COMPARE_SIZE = (640, 480)


# ROI (coordenadas en píxeles)
ROI_POLYGON = np.array([
   [92, 5],
   [216, 711],
   [1040, 716],
   [939, 238],
   [624, 165],
   [102, 11]
], np.int32).reshape((-1, 1, 2))
STRICT_ROI = True  # True = todas dentro, False = basta una


# Depuración
SAVE_DEBUG = True


# =======================
# CARGAR MODELOS
# =======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO(YOLO_PATH)
sam_model = SAM(SAM_PATH)  # Ultralytics SAM2


# =======================
# FUNCIONES
# =======================
def detect_cows_and_mask(frame):
   """
   Detecta vacas con YOLO y segmenta con SAM2 (Ultralytics).
   """
   img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   results = yolo_model.predict(img_rgb, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
   res = results[0]
   names = res.names


   mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)
   detections = []


   if res.boxes is None or len(res.boxes) == 0:
       return mask_total, detections


   xyxy = res.boxes.xyxy.cpu().numpy()
   cls_ids = res.boxes.cls.cpu().numpy().astype(int)
   confs = res.boxes.conf.cpu().numpy()


   for (box, cls_id, conf) in zip(xyxy, cls_ids, confs):
       class_name = names[int(cls_id)].lower()
       if class_name not in TARGET_CLASSES:
           continue


       x1, y1, x2, y2 = map(int, box.tolist())
       bbox = [x1, y1, x2, y2]


       # SAM2 con bbox (sin multimask)
       try:
           sam_out = sam_model.predict(img_rgb, bboxes=[bbox])
       except Exception as e:
           print("⚠️ Error con SAM2:", e)
           continue


       # Extraer máscara
       mask = None
       try:
           if sam_out and hasattr(sam_out[0], "masks"):
               mask = sam_out[0].masks.data[0].cpu().numpy().astype(np.uint8)
       except Exception:
           mask = None


       if mask is None:
           continue


       mask = (mask > 0).astype(np.uint8)
       mask_total = np.logical_or(mask_total, mask).astype(np.uint8)


       detections.append({
           "class": class_name,
           "score": float(conf),
           "bbox": bbox,
           "mask_area_px": int(mask.sum())
       })


   return mask_total, detections




def check_roi(detections, roi_polygon, strict=True):
   """
   Verifica si las vacas están dentro del ROI.
   """
   results = []
   for det in detections:
       x1, y1, x2, y2 = det["bbox"]
       cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
       inside = cv2.pointPolygonTest(roi_polygon, (cx, cy), False)
       results.append(inside >= 0)
   return all(results) if strict else any(results)




# =======================
# PIPELINE DE VIDEO
# =======================
def process_video(video_path, output_dir):
   os.makedirs(output_dir, exist_ok=True)
   mask_dir = os.path.join(output_dir, "masks", os.path.splitext(os.path.basename(video_path))[0])
   img_dir = os.path.join(output_dir, "images", os.path.splitext(os.path.basename(video_path))[0])
   ann_dir = os.path.join(output_dir, "annotations", os.path.splitext(os.path.basename(video_path))[0])
   debug_dir = os.path.join(output_dir, "debug")
   os.makedirs(mask_dir, exist_ok=True)
   os.makedirs(img_dir, exist_ok=True)
   os.makedirs(ann_dir, exist_ok=True)
   if SAVE_DEBUG: os.makedirs(debug_dir, exist_ok=True)


   video_name = os.path.splitext(os.path.basename(video_path))[0]
   cap = cv2.VideoCapture(video_path)
   fps = cap.get(cv2.CAP_PROP_FPS) or 30.0


   fgbg = cv2.createBackgroundSubtractorMOG2(
       history=MOG_HISTORY,
       varThreshold=MOG_VAR_THRESHOLD,
       detectShadows=MOG_DETECT_SHADOWS
   )


   frame_count = 0
   saved_count = 0
   start_time = time.time()
   last_saved_frame = None


   while True:
       ret, frame = cap.read()
       if not ret:
           break


       frame_count += 1
       if frame_count % FRAME_SKIP != 0:
           continue


       # ROI + MOG2
       mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
       cv2.fillPoly(mask_roi, [ROI_POLYGON], 255)
       frame_roi = cv2.bitwise_and(frame, frame, mask=mask_roi)


       gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
       fgmask_raw = fgbg.apply(gray)


       _, fgmask = cv2.threshold(fgmask_raw, 200, 255, cv2.THRESH_BINARY)
       kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
       fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
       fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)


       contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       movement_detected = any(cv2.contourArea(cnt) > MIN_AREA for cnt in contours)


       if not movement_detected:
           continue


       # YOLO + SAM2
       mask_total, detections = detect_cows_and_mask(frame)


       if len(detections) > 0 and check_roi(detections, ROI_POLYGON, strict=STRICT_ROI):
           frame_name = f"{video_name}_frame_{frame_count:06d}.jpg"
           mask_name = f"{video_name}_frame_{frame_count:06d}_mask.png"
           ann_name = f"{video_name}_frame_{frame_count:06d}.json"


           img_path = os.path.join(img_dir, frame_name)
           mask_path = os.path.join(mask_dir, mask_name)
           ann_path = os.path.join(ann_dir, ann_name)


           # Deduplicación por diferencia de imagen
           should_save = True
           if last_saved_frame is not None:
               prev = cv2.resize(last_saved_frame, DUPLICATE_COMPARE_SIZE)
               curr = cv2.resize(frame, DUPLICATE_COMPARE_SIZE)
               diff = np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32)))
               if diff < DUPLICATE_SAVE_DIFF:
                   should_save = False

           if should_save:
               cv2.imwrite(img_path, frame)
               Image.fromarray((mask_total * 255).astype(np.uint8)).save(mask_path)


               annotation = {
                   "image_id": f"{video_name}_frame_{frame_count:06d}",
                   "file_name": f"images/{video_name}/{frame_name}",
                   "mask_path": f"masks/{video_name}/{mask_name}",
                   "video_path": os.path.basename(video_path),
                   "frame_index_in_video": frame_count,
                   "timestamp_sec_in_video": round(frame_count / fps, 2),
                   "timestamp_saved_utc": datetime.now(timezone.utc).isoformat(),
                   "detections": detections
               }


               with open(ann_path, "w") as f:
                   json.dump(annotation, f, indent=2)


               last_saved_frame = frame.copy()
               saved_count += 1
               print(f"✅ Guardado frame: {frame_name}")


               if SAVE_DEBUG:
                   dbg = frame.copy()
                   cv2.polylines(dbg, [ROI_POLYGON], True, (0,255,255), 2)
                   for det in detections:
                       x1,y1,x2,y2 = det["bbox"]
                       cv2.rectangle(dbg, (x1,y1),(x2,y2),(0,255,0),2)
                   cv2.imwrite(os.path.join(debug_dir, f"debug_{frame_name}"), dbg)


   cap.release()
   elapsed = time.time() - start_time
   print("\n=== Resumen ===")
   print(f"Frames procesados: {frame_count}")
   print(f"Frames guardados con anotaciones: {saved_count}")
   print(f"Tiempo total de ejecución: {elapsed:.2f} segundos")




if __name__ == "__main__":
   process_video(VIDEO_PATH, OUTPUT_DIR)
