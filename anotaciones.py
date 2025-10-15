"""
Script simple para añadir campos faltantes a las anotaciones JSON.
Preserva toda la información existente.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm


# =======================
# CONFIGURACIÓN - EDITAR AQUÍ
# =======================

# Directorio base donde están las anotaciones
BASE_DIR = r"/home/pipe/Documentos/Proyecto_Ganado/annotations/0407"

# Parámetros a añadir (editar según tus necesidades)
PARAMS = {
    "video_id": "0407",                    # ID del video procesado
    "camera_id": "ch4_main",               # ID de la cámara
    "image_width": 1920,                   # Ancho de las imágenes
    "image_height": 1080,                  # Alto de las imágenes
    "annotator": "auto_v1",                # Quién/qué generó las anotaciones
    "annotation_version": "v1.0"           # Versión del esquema de anotación
}

# Corrección de video_path (opcional)
# Si es None, mantiene el video_path original
# Si defines un valor, lo reemplaza
CORRECT_VIDEO_PATH = None  # Ejemplo: "raw_videos/2025-03-04/0407.mp4"


# =======================
# FUNCIÓN PRINCIPAL
# =======================

def add_fields_to_json(json_path, params, correct_video_path=None):
    """
    Añade campos faltantes a un archivo JSON de anotación.
    
    Args:
        json_path: Ruta al archivo JSON
        params: Diccionario con los parámetros a añadir
        correct_video_path: Nueva ruta para video_path (opcional)
    """
    try:
        # Leer JSON existente
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Añadir campos nuevos (solo si no existen)
        for key, value in params.items():
            if key not in data:
                data[key] = value
        
        # Corregir video_path si se especificó
        if correct_video_path is not None:
            data['video_path'] = correct_video_path
        
        # Reorganizar campos en orden específico
        ordered_data = {}
        
        # Orden deseado de campos
        field_order = [
            'image_id',
            'file_name',
            'mask_path',
            'video_path',
            'video_id',
            'camera_id',
            'frame_index_in_video',
            'timestamp_sec_in_video',
            'timestamp_saved_utc',
            'image_width',
            'image_height',
            'annotator',
            'annotation_version',
            'detections'
        ]
        
        # Primero añadir campos en el orden especificado
        for field in field_order:
            if field in data:
                ordered_data[field] = data[field]
        
        # Luego añadir cualquier campo adicional que no esté en el orden
        for key, value in data.items():
            if key not in ordered_data:
                ordered_data[key] = value
        
        # Guardar JSON actualizado con orden
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ordered_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"❌ Error en {json_path}: {e}")
        return False


def process_all_annotations(base_dir, params, correct_video_path=None):
    """
    Procesa todos los archivos JSON en el directorio de anotaciones.
    
    Args:
        base_dir: Directorio base del dataset
        params: Diccionario con los parámetros a añadir
        correct_video_path: Nueva ruta para video_path (opcional)
    """
    # Buscar archivos JSON directamente desde base_dir (busca recursivamente)
    if not os.path.exists(base_dir):
        print(f"❌ No se encontró el directorio: {base_dir}")
        return
    
    # Encontrar todos los archivos JSON recursivamente
    json_files = list(Path(base_dir).rglob("*.json"))
    
    if not json_files:
        print("❌ No se encontraron archivos JSON")
        return
    
    print(f"📁 Encontrados {len(json_files)} archivos JSON")
    print(f"🔧 Campos a añadir: {list(params.keys())}")
    if correct_video_path:
        print(f"📝 Corrigiendo video_path a: {correct_video_path}")
    print()
    
    # Procesar cada archivo
    success = 0
    failed = 0
    
    for json_path in tqdm(json_files, desc="Procesando"):
        if add_fields_to_json(str(json_path), params, correct_video_path):
            success += 1
        else:
            failed += 1
    
    # Resumen
    print(f"\n{'='*50}")
    print(f"✅ Procesados correctamente: {success}")
    print(f"❌ Con errores: {failed}")
    print(f"{'='*50}")


# =======================
# EJECUCIÓN
# =======================

if __name__ == "__main__":
    print("="*50)
    print("AÑADIR CAMPOS A ANOTACIONES JSON")
    print("="*50)
    print(f"\nDirectorio base: {BASE_DIR}")
    print(f"\nParámetros a añadir:")
    for key, value in PARAMS.items():
        print(f"  - {key}: {value}")
    print()
    
    # Confirmar antes de ejecutar
    confirm = input("¿Continuar? (s/n): ")
    if confirm.lower() == 's':
        process_all_annotations(BASE_DIR, PARAMS, CORRECT_VIDEO_PATH)
    else:
        print("Operación cancelada")