#!/usr/bin/env python3
"""
convert_and_remove_h264.py


Convierte recursivamente archivos .h264/.264/.bin -> .mp4 manteniendo la estructura.
Borra el archivo original solo si la conversión fue exitosa (output existe y tamaño > 0).


Uso (in-place, por defecto borra originales):
 python convert_and_remove_h264.py /home/pipe/Documentos/Proyecto_Ganado/16feb2025/XVR_ch3_main_20250216103405_20250216103548.dav


Ejemplo (convertir en otro dir y eliminar originales si conversión OK):
 python convert_and_remove_h264.py /ruta/a/videos_root --output_dir /ruta/a/mp4_output


Flags:
 --no_reencode_on_fail   : si remux falla, NO intentar recodificar (mayor compatibilidad)
 --ffmpeg_cmd PATH       : ruta a ffmpeg si no está en PATH
 --dry_run               : muestra acciones sin ejecutar conversiones ni borrados
"""
import sys
import subprocess
from pathlib import Path
import argparse


H264_EXTS = {".h264", ".264", ".bin", ".dav"}


def run_cmd(cmd):
   res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
   return res.returncode, res.stdout, res.stderr


def ffmpeg_remux(in_path: Path, out_path: Path, ffmpeg_cmd="ffmpeg"):
   cmd = [ffmpeg_cmd, "-y", "-i", str(in_path), "-c:v", "copy", "-c:a", "aac", str(out_path)]
   code, out, err = run_cmd(cmd)
   return code == 0, err


def ffmpeg_reencode(in_path: Path, out_path: Path, ffmpeg_cmd="ffmpeg"):
   cmd = [ffmpeg_cmd, "-y", "-i", str(in_path), "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", str(out_path)]
   code, out, err = run_cmd(cmd)
   return code == 0, err


def is_valid_file(p: Path):
   try:
       return p.exists() and p.is_file() and p.stat().st_size > 0
   except Exception:
       return False


def convert_tree(root_dir: Path, output_dir: Path = None, reencode_on_fail: bool = True,
                ffmpeg_cmd: str = "ffmpeg", dry_run: bool = False):
   root_dir = root_dir.expanduser().resolve()
   if not root_dir.is_dir():
       raise SystemExit(f"Ruta no válida: {root_dir}")


   matches = list(root_dir.rglob("*"))
   # contar mp4 y h264
   mp4_files = [p for p in matches if p.is_file() and p.suffix.lower() == ".mp4"]
   h264_files = [p for p in matches if p.is_file() and p.suffix.lower() in H264_EXTS]
   print(f"Encontrados {len(mp4_files)} archivos .mp4 y {len(h264_files)} archivos h264-type en {root_dir}")


   stats = {"converted":0, "skipped_exist_mp4":0, "deleted_originals":0, "failed":0, "skipped_no_action":0}


   for f in h264_files:
       # determinar ruta de salida respetando estructura relativa si se indicó output_dir
       if output_dir:
           try:
               rel = f.parent.relative_to(root_dir)
           except Exception:
               rel = Path(".")
           out_parent = (Path(output_dir).expanduser().resolve() / rel)
           out_parent.mkdir(parents=True, exist_ok=True)
           out_mp4 = out_parent / (f.stem + ".mp4")
       else:
           out_mp4 = f.with_suffix(".mp4")


       # Si ya existe el mp4 de destino: omitimos convertir, pero BORRAMOS original si mp4 es válido
       if is_valid_file(out_mp4):
           print(f"[exists] mp4 ya existe: {out_mp4}")
           stats["skipped_exist_mp4"] += 1
           # borrar original para dejar solo mp4 (si la salida es válida)
           try:
               print(f"  -> eliminando original: {f}")
               if not dry_run:
                   f.unlink()
               stats["deleted_originals"] += 1
           except Exception as e:
               print(f"  ! no se pudo borrar original {f}: {e}")
           continue


       print(f"[convert] {f} -> {out_mp4}")
       if dry_run:
           print("  (dry run) no se ejecuta ffmpeg")
           stats["skipped_no_action"] += 1
           continue


       # Intentar remux (rápido)
       ok, err = ffmpeg_remux(f, out_mp4, ffmpeg_cmd=ffmpeg_cmd)
       if ok and is_valid_file(out_mp4):
           print("  remux OK")
           stats["converted"] += 1
           # borrar original
           try:
               f.unlink()
               stats["deleted_originals"] += 1
               print("  original eliminado")
           except Exception as e:
               print(f"  ! no se pudo borrar original {f}: {e}")
           continue


       # Si remux falla, intentar recodificar si se permite
       print("  remux falló.")
       if reencode_on_fail:
           print("  intentando recodificar con libx264...")
           ok2, err2 = ffmpeg_reencode(f, out_mp4, ffmpeg_cmd=ffmpeg_cmd)
           if ok2 and is_valid_file(out_mp4):
               print("  reencode OK")
               stats["converted"] += 1
               # borrar original
               try:
                   f.unlink()
                   stats["deleted_originals"] += 1
                   print("  original eliminado")
               except Exception as e:
                   print(f"  ! no se pudo borrar original {f}: {e}")
               continue
           else:
               print(f"  ERROR: no se pudo convertir {f}. ffmpeg output (trunc):")
               print((err2 or err).splitlines()[:6])
               # eliminar archivo de salida parcial si existe
               try:
                   if out_mp4.exists():
                       out_mp4.unlink()
               except Exception:
                   pass
               stats["failed"] += 1
       else:
           print("  reencode deshabilitado (no_reencode_on_fail). marcado como fallido.")
           stats["failed"] += 1


   # resumen
   print("\n=== Resumen ===")
   for k,v in stats.items():
       print(f"  {k}: {v}")


def check_ffmpeg(path="ffmpeg"):
   try:
       code, out, err = run_cmd([path, "-version"])
       return code == 0
   except FileNotFoundError:
       return False


def run_cmd(cmd):
   res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
   return res.returncode, res.stdout, res.stderr


if __name__ == "__main__":
   p = argparse.ArgumentParser(description="Convierte h264->mp4 y elimina originales si la conversión fue exitosa.")
   p.add_argument("root_dir", help="Carpeta raíz que contiene subcarpetas por fecha con los videos.")
   p.add_argument("--output_dir", help="(opcional) carpeta donde crear los .mp4 respetando la estructura relativa.")
   p.add_argument("--no_reencode_on_fail", dest="reencode_on_fail", action="store_false",
                  help="Si remux falla, no intentar recodificar (por defecto sí recodifica).")
   p.add_argument("--ffmpeg_cmd", default="ffmpeg", help="Comando ffmpeg (ruta si no está en PATH).")
   p.add_argument("--dry_run", action="store_true", help="Mostrar acciones sin ejecutar conversiones ni borrados.")
   args = p.parse_args()


   if not check_ffmpeg(args.ffmpeg_cmd):
       print("ffmpeg no encontrado o no funcional en la ruta proporcionada. Instala ffmpeg o pasa --ffmpeg_cmd /ruta/ffmpeg")
       sys.exit(1)


   convert_tree(Path(args.root_dir),
                output_dir=Path(args.output_dir) if args.output_dir else None,
                reencode_on_fail=args.reencode_on_fail,
                ffmpeg_cmd=args.ffmpeg_cmd,
                dry_run=args.dry_run)
