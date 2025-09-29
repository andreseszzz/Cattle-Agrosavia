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
VIDEO_PATH = r"/home/pipe/Documentos/Proyecto_Ganado/Finca_Tibaitata/a5-20250924T204658Z-1-003/a5/XVR_ch5_main_20250825135308_20250825140320.mp4"  # <--- Carpeta con los videos a procesar
OUTPUT_DIR = r"/home/pipe/Documentos/Proyecto_Ganado/Finca_Tibaitata/a5-20250924T204658Z-1-003"
SAM_TYPE = "vit_h"
SAM_PATH = r"V1.1/muestre_gradiente_MOG2/sam_vit_h_4b8939.pth"
YOLO_PATH = r"V1.1/muestre_gradiente_MOG2/yolo12m.pt"


# Detección / muestreo
FRAME_SKIP = 60                # ajustar según lo conversado (ej: 30 -> ~1 fps si video 30fps)
MIN_AREA = 4000                # área mínima (aumentar para ignorar pasto/sombra)
# MOG2 params
MOG_HISTORY = 1000
MOG_VAR_THRESHOLD = 50
MOG_DETECT_SHADOWS = True


# Duplicados (comparación por diferencia media en versión pequeña)
# Pon None para desactivar; si quieres activar la deduplicación, usa un valor pequeño (ej 3-6)
DUPLICATE_SAVE_DIFF = 3.0
DUPLICATE_COMPARE_SIZE = (320, 240)  # tamaño para comparar (más pequeño => más rápido)


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
   # results[0] es el primer resultado por frame
   boxes = results[0].boxes
   names = results[0].names


   mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)
   detections = []


   # set_image una sola vez por frame (más eficiente)
   sam_predictor.set_image(img_rgb)


   # Iterar sobre cajas (si no hay cajas, boxes puede estar vacío)
   for box in boxes:
       # Extracción segura de class id
       try:
           class_id = int(box.cls[0])
       except Exception:
           class_id = int(box.cls)


       class_name = names[class_id]
       if class_name.lower() != "cow":
           continue


       # Obtener coordenadas
       x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
       bbox = [int(x1), int(y1), int(x2), int(y2)]


       # Preparar input para SAM (forma: [ [x1,y1,x2,y2] ])
       input_box = np.array([bbox])
       masks, scores_sam, _ = sam_predictor.predict(
           box=input_box,
           multimask_output=False
       )


       if masks is None or len(masks) == 0:
           continue


       mask = masks[0].astype(np.uint8)
       mask_total = np.logical_or(mask_total, mask).astype(np.uint8)


       detections.append({
           "class": class_name,
           "score": float(scores_sam[0]) if (scores_sam is not None and len(scores_sam)>0) else None,
           "bbox": bbox,
           "mask_area_px": int(mask.sum())
       })


   return mask_total, detections




# =======================
# PROCESAMIENTO DE VIDEO (MOG2)
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
   fps = cap.get(cv2.CAP_PROP_FPS) or 30.0


   # Iniciar sustractor MOG2
   fgbg = cv2.createBackgroundSubtractorMOG2(history=MOG_HISTORY,
                                             varThreshold=MOG_VAR_THRESHOLD,
                                             detectShadows=MOG_DETECT_SHADOWS)


   frame_count = 0
   saved_count = 0
   start_time = time.time()


   # Para deduplicación
   last_saved_small = None


   while True:
       ret, frame = cap.read()
       if not ret:
           break


       frame_count += 1


       # Convertir a gris (usado por MOG2 y deduplicación)
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


       # Aplicar MOG2 en **cada** frame para mantener actualizado el modelo de fondo
       fgmask_raw = fgbg.apply(gray)


       # Si no toca procesar (FRAME_SKIP), sólo seguimos (pero MOG2 ya se actualizó)
       if frame_count % FRAME_SKIP != 0:
           continue


       # Limpiar máscara: eliminar sombras (valor 127) y ruido
       # Umbral para quitar sombras (si detectShadows=True, sombras ~127)
       _, fgmask = cv2.threshold(fgmask_raw, 200, 255, cv2.THRESH_BINARY)
       kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
       fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
       fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)


       # Encontrar contornos en la máscara procesada
       contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       movement_detected = any(cv2.contourArea(cnt) > MIN_AREA for cnt in contours)


       if movement_detected:
           # === Opcional: deduplicación simple antes de hacer YOLO+SAM ===
           if DUPLICATE_SAVE_DIFF is not None:
               # comparar versión pequeña para mayor velocidad
               small = cv2.resize(gray, DUPLICATE_COMPARE_SIZE, interpolation=cv2.INTER_LINEAR)
               if last_saved_small is not None:
                   mad = np.mean(np.abs(small.astype(np.int16) - last_saved_small.astype(np.int16)))
                   if mad < DUPLICATE_SAVE_DIFF:
                       # Muy similar al último guardado -> omitir para evitar repetidos
                       print(f"⚠️ Frame {frame_count} muy similar al último guardado (MAD={mad:.2f}), omitiendo...")
                       continue


           # Llamar a YOLO + SAM (costoso)
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


               # Actualizar referencia para deduplicación
               if DUPLICATE_SAVE_DIFF is not None:
                   last_saved_small = cv2.resize(gray, DUPLICATE_COMPARE_SIZE, interpolation=cv2.INTER_LINEAR)


   cap.release()
   elapsed = time.time() - start_time


   print("\n=== Resumen ===")
   print(f"Frames procesados: {frame_count}")
   print(f"Frames guardados con anotaciones: {saved_count}")
   print(f"Tiempo total de ejecución: {elapsed:.2f} segundos")




if __name__ == "__main__":
   process_video(VIDEO_PATH, OUTPUT_DIR)
