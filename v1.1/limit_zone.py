import cv2
import numpy as np


points = []


def click_event(event, x, y, flags, param):
   if event == cv2.EVENT_LBUTTONDOWN:
       points.append((x, y))
       print(f"{x,y}")


frame = cv2.imread("/home/pipe/Documentos/Proyecto_Ganado/Finca_San_Alberto/04-03-2025/XVR_ch4_main_20250304060000_20250304070000/images/XVR_ch4_main_20250304060000_20250304070000/XVR_ch4_main_20250304060000_20250304070000_frame_000010.jpg")  # un frame del video
cv2.imshow("Selecciona ROI", frame)
cv2.setMouseCallback("Selecciona ROI", click_event)


cv2.waitKey(0) 
cv2.destroyAllWindows()


print("Coordenadas ROI:", points)
