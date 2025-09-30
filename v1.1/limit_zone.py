import cv2
import numpy as np


points = []


def click_event(event, x, y, flags, param):
   if event == cv2.EVENT_LBUTTONDOWN:
       points.append((x, y))
       print(f"Punto agregado: {x,y}")


frame = cv2.imread("/home/pipe/Documentos/Proyecto_Ganado/Finca_San_Alberto/05-03-2025/709.jpg")  # un frame del video
cv2.imshow("Selecciona ROI", frame)
cv2.setMouseCallback("Selecciona ROI", click_event)


cv2.waitKey(0) 
cv2.destroyAllWindows()


print("Coordenadas ROI:", points)
