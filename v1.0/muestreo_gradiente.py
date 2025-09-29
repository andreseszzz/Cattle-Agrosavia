import cv2
import os
import numpy as np
import json
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from PIL import Image
import torch
import time
from datetime import datetime, timezone


# =======================
# CONFIGURACIONES
# =======================
VIDEO_PATH = r"/home/pipe/Documentos/yolov11/muestreo_gradiente/clip3_to_Mold.mp4"
OUTPUT_DIR = r"/home/pipe/Documentos/yolov11/muestreo_gradiente"
SAM_TYPE = "vit_h"
SAM_PATH = r"muestreo_gradiente/sam_vit_h_4b8939.pth"
YOLO_PATH = r"/home/pipe/Documentos/yolov11/yolo11m.pt"


FRAME_DIFF_THRESHOLD = 30
MIN_AREA = 500
FRAME_SKIP = 2


# =======================
# CARGAR MODELOS
# =======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_PATH)
sam.to(DEVICE)
sam_predictor = SamPredictor(sam)
yolo_model = YOLO(YOLO_PATH)


def detect_cows_and_mask(frame):
   """
   Detecta vacas con YOLO y segmenta con SAM.
   Devuelve una máscara combinada y lista de detecciones.
   """
   img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   results = yolo_model(img_rgb)
   boxes = results[0].boxes
   names = results[0].names


   mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)
   detections = []


   for box in boxes:
       class_id = int(box.cls[0])
       class_name = names[class_id]


       if class_name.lower() != "cow":
           continue


       x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
       bbox = [int(x1), int(y1), int(x2), int(y2)]


       sam_predictor.set_image(img_rgb)
       input_box = np.array([bbox])
       masks, scores_sam, _ = sam_predictor.predict(
           box=input_box,
           multimask_output=False
       )
       mask = masks[0].astype(np.uint8)
       mask_total = np.logical_or(mask_total, mask).astype(np.uint8)


       detections.append({
           "class": class_name,
           "score": float(scores_sam[0]),
           "bbox": bbox,
           "mask_area_px": int(mask.sum())
       })


   return mask_total, detections




# =======================
# PROCESAMIENTO DE VIDEO
# =======================
def process_video(video_path, output_dir):
   os.makedirs(output_dir, exist_ok=True)
   mask_dir = os.path.join(output_dir, "masks", os.path.splitext(os.path.basename(video_path))[0])
   img_dir = os.path.join(output_dir, "images", os.path.splitext(os.path.basename(video_path))[0])
   ann_dir = os.path.join(output_dir, "annotations", os.path.splitext(os.path.basename(video_path))[0])
   os.makedirs(mask_dir, exist_ok=True)
   os.makedirs(img_dir, exist_ok=True)
   os.makedirs(ann_dir, exist_ok=True)


   video_name = os.path.splitext(os.path.basename(video_path))[0]
   cap = cv2.VideoCapture(video_path)
   fps = cap.get(cv2.CAP_PROP_FPS)


   ret, prev_frame = cap.read()
   if not ret:
       print("❌ No se pudo leer el video.")
       return


   prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
   frame_count = 0
   saved_count = 0


   start_time = time.time()


   while True:
       ret, frame = cap.read()
       if not ret:
           break


       frame_count += 1
       if frame_count % FRAME_SKIP != 0:
           continue


       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       diff = cv2.absdiff(prev_gray, gray)
       _, thresh = cv2.threshold(diff, FRAME_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)


       contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       movement_detected = any(cv2.contourArea(cnt) > MIN_AREA for cnt in contours)


       if movement_detected:
           mask_total, detections = detect_cows_and_mask(frame)
           if len(detections) > 0:
               frame_name = f"{video_name}_frame_{frame_count:06d}.jpg"
               mask_name = f"{video_name}_frame_{frame_count:06d}_mask.png"
               ann_name = f"{video_name}_frame_{frame_count:06d}.json"


               img_path = os.path.join(img_dir, frame_name)
               mask_path = os.path.join(mask_dir, mask_name)
               ann_path = os.path.join(ann_dir, ann_name)


               # Guardar imagen y máscara
               cv2.imwrite(img_path, frame)
               Image.fromarray((mask_total * 255).astype(np.uint8)).save(mask_path)


               # Construir anotación
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


               saved_count += 1
               print(f"✅ Guardado frame: {frame_name}, máscara: {mask_name}, anotación: {ann_name}")


       prev_gray = gray


   cap.release()
   elapsed = time.time() - start_time


   print("\n=== Resumen ===")
   print(f"Frames procesados: {frame_count}")
   print(f"Frames guardados con anotaciones: {saved_count}")
   print(f"Tiempo total de ejecución: {elapsed:.2f} segundos")




if __name__ == "__main__":
   process_video(VIDEO_PATH, OUTPUT_DIR)
