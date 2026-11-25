"""
generar_videos_test.py
======================
Script auxiliar para pre-generar videos de press banca anotados con YOLOv8-pose.

Uso como CLI:
    python generar_videos_test.py --base-dir ./data --salida-dir ./videos_anotados

Función exportable para uso desde app_dashboard.py:
    from generar_videos_test import anotar_video_pose
"""
from typing import Any, Optional

imageio_ffmpeg: Any = None
try:
    import imageio_ffmpeg  # type: ignore[no-redef]
except ImportError:
    imageio_ffmpeg = None
import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración de YOLOv8-pose
# ---------------------------------------------------------------------------

YOLO_MODEL_NAME = "yolov8n-pose.pt"
YOLO_CONF = 0.70

# Conexiones del esqueleto COCO-17 keypoints
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # cabeza / ojos / orejas
    (5, 6),                                     # hombros
    (5, 7), (7, 9),                             # brazo izquierdo
    (6, 8), (8, 10),                            # brazo derecho
    (5, 11), (6, 12),                           # torso superior
    (11, 12),                                   # cadera
    (11, 13), (13, 15),                         # pierna izquierda
    (12, 14), (14, 16),                         # pierna derecha
]

# Colores por conexión (BGR)
COLORES_CONEXION = [
    (255, 128,   0), (255, 128,   0), (255, 128,   0), (255, 128,   0),
    ( 51, 153, 255),
    ( 51, 153, 255), ( 51, 153, 255),
    ( 51, 153, 255), ( 51, 153, 255),
    (  0, 220,   0), (  0, 220,   0),
    (  0, 220,   0),
    (  0, 220,   0), (  0, 220,   0),
    (  0, 220,   0), (  0, 220,   0),
]

# Colores del cartel por clase (BGR)
COLORES_CLASE_BGR = {
    "tecnica_correcta":     ( 46, 204,  46),
    "codos_abiertos":       ( 39, 174, 230),
    "rebote_en_pecho":      ( 46,  46, 230),
    "empuje_desbalanceado": (146,  46, 160),
}


# ---------------------------------------------------------------------------
# Verificación de dependencias externas
# ---------------------------------------------------------------------------

def _obtener_ffmpeg_exe() -> Optional[str]:
    """Devuelve el ejecutable de ffmpeg del sistema o el de imageio-ffmpeg."""
    cmd_sistema = shutil.which("ffmpeg")
    if cmd_sistema:
        return cmd_sistema
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return None


def _verificar_ffmpeg() -> bool:
    """Devuelve True si ffmpeg está disponible en el sistema o vía imageio-ffmpeg."""
    return _obtener_ffmpeg_exe() is not None


def _verificar_ultralytics() -> bool:
    """Devuelve True si ultralytics está instalado."""
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Funciones de dibujo
# ---------------------------------------------------------------------------

def _dibujar_esqueleto(
    frame: np.ndarray,
    kps_norm: np.ndarray,
    kps_conf: np.ndarray,
) -> np.ndarray:
    """
    Dibuja keypoints y conexiones del esqueleto COCO sobre el frame.

    Parameters
    ----------
    frame     : BGR image (H, W, 3)
    kps_norm  : keypoints normalizados, shape (17, 2) — valores en [0, 1]
    kps_conf  : confianza por keypoint, shape (17,)
    """
    h, w = frame.shape[:2]
    CONF_MIN = 0.30

    # Conexiones
    for idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
        if i >= len(kps_norm) or j >= len(kps_norm):
            continue
        ci = float(kps_conf[i]) if i < len(kps_conf) else 0.0
        cj = float(kps_conf[j]) if j < len(kps_conf) else 0.0
        if ci < CONF_MIN or cj < CONF_MIN:
            continue
        x1 = int(kps_norm[i][0] * w)
        y1 = int(kps_norm[i][1] * h)
        x2 = int(kps_norm[j][0] * w)
        y2 = int(kps_norm[j][1] * h)
        color = COLORES_CONEXION[idx] if idx < len(COLORES_CONEXION) else (200, 200, 0)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Keypoints
    for kp, cf in zip(kps_norm, kps_conf):
        if float(cf) < CONF_MIN:
            continue
        x, y = int(kp[0] * w), int(kp[1] * h)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (30, 30, 30),   1,  cv2.LINE_AA)

    return frame


def _dibujar_cartel(
    frame: np.ndarray,
    clase_real: str,
    pred_rf: str,
    prob_rf: float,
) -> np.ndarray:
    """Dibuja el cartel semitransparente en la parte superior del frame."""
    h, w = frame.shape[:2]
    altura_cartel = 65

    # Fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, altura_cartel), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    color_real = COLORES_CLASE_BGR.get(clase_real, (200, 200, 200))
    color_pred = COLORES_CLASE_BGR.get(pred_rf, (200, 200, 200))

    escala = max(0.42, w / 1280.0 * 0.68)
    grosor  = 1

    etiq_real = f"Real: {clase_real.replace('_', ' ').title()}"
    etiq_pred = f"Pred RF: {pred_rf.replace('_', ' ').title()} ({prob_rf * 100:.1f}%)"

    cv2.putText(frame, etiq_real, (10, 24),
                cv2.FONT_HERSHEY_DUPLEX, escala, color_real, grosor, cv2.LINE_AA)
    cv2.putText(frame, etiq_pred, (10, 52),
                cv2.FONT_HERSHEY_DUPLEX, escala, color_pred, grosor, cv2.LINE_AA)

    # Icono de acierto/error
    acierto = clase_real.lower() == pred_rf.lower()
    icono   = chr(10003) if acierto else chr(10007)   # ✓ / ✗ ASCII seguro
    color_i = (0, 200, 0) if acierto else (0, 0, 220)
    cv2.putText(frame, icono, (w - 35, 42),
                cv2.FONT_HERSHEY_DUPLEX, escala * 1.3, color_i, 2, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Función principal exportable
# ---------------------------------------------------------------------------

def anotar_video_pose(
    video_path: str,
    salida_dir: str,
    clase_real: str,
    pred_rf: str,
    prob_rf: float = 1.0,
    prefijo: str = "web_",
    conf: float = YOLO_CONF,
) -> Optional[str]:
    """
    Aplica YOLOv8-pose al video y genera una versión anotada lista para el navegador.

    Parameters
    ----------
    video_path  : ruta al video MP4 original.
    salida_dir  : carpeta de salida.
    clase_real  : clase real de la toma (para el cartel).
    pred_rf     : predicción del Random Forest (para el cartel).
    prob_rf     : probabilidad de la predicción (0–1).
    prefijo     : prefijo del archivo de salida (default "web_").
    conf        : umbral de confianza YOLOv8 (default 0.70).

    Returns
    -------
    Ruta al video anotado (MP4 recodificado o AVI si ffmpeg no está instalado),
    o None si ocurrió un error no recuperable.
    """
    if not _verificar_ultralytics():
        logger.error(
            "ultralytics no está instalado. Instálalo con: pip install ultralytics"
        )
        return None

    from ultralytics import YOLO  # importación diferida para evitar error si no está

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        logger.error("Video no encontrado: %s", video_path)
        return None

    os.makedirs(salida_dir, exist_ok=True)

    nombre_base  = video_path_obj.stem
    salida_tmp   = Path(salida_dir) / f"_tmp_{nombre_base}.avi"
    salida_final = Path(salida_dir) / f"{prefijo}{nombre_base}.mp4"

    # --- Cargar modelo ---
    try:
        modelo = YOLO(YOLO_MODEL_NAME)
        logger.info("Modelo cargado: %s  (conf=%.2f)", YOLO_MODEL_NAME, conf)
    except Exception as exc:
        logger.error("No se pudo cargar %s: %s", YOLO_MODEL_NAME, exc)
        return None

    # --- Abrir video ---
    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        logger.error("No se pudo abrir el video: %s", video_path)
        return None

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(salida_tmp), fourcc, fps, (ancho, alto))

    n_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inferencia
            results = modelo(frame, conf=conf, verbose=False)

            for result in results:
                if result.keypoints is not None and len(result.keypoints.data) > 0:
                    kps = result.keypoints
                    xy_norm = kps.xyn[0].cpu().numpy()   # (17, 2) normalizado
                    conf_kp = (
                        kps.conf[0].cpu().numpy()
                        if kps.conf is not None
                        else np.ones(len(xy_norm))
                    )
                    frame = _dibujar_esqueleto(frame, xy_norm, conf_kp)

            frame = _dibujar_cartel(frame, clase_real, pred_rf, prob_rf)
            writer.write(frame)
            n_frames += 1

    finally:
        cap.release()
        writer.release()

    logger.info("Frames procesados: %d", n_frames)

    # --- Recodificar con ffmpeg ---
    ffmpeg_exe = _obtener_ffmpeg_exe()
    if ffmpeg_exe:
        try:
            cmd = [
                ffmpeg_exe, "-y",
                "-i",          str(salida_tmp),
                "-vcodec",     "libx264",
                "-pix_fmt",    "yuv420p",
                "-preset",     "fast",
                "-crf",        "23",
                str(salida_final),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(salida_tmp)
            logger.info("Video recodificado a H.264: %s", salida_final)
            return str(salida_final)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            logger.warning("ffmpeg falló: %s — se devuelve AVI sin recodificar.", stderr[:200])
    else:
        logger.warning(
            "ffmpeg no encontrado en PATH ni en imageio-ffmpeg. El video se guardó como AVI y puede no "
            "reproducirse en el navegador. Instala ffmpeg: https://ffmpeg.org/download.html"
        )

    # Guardar AVI como resultado alternativo
    salida_avi = Path(salida_dir) / f"{prefijo}{nombre_base}.avi"
    try:
        os.rename(salida_tmp, salida_avi)
    except OSError:
        pass
    return str(salida_avi)


# ---------------------------------------------------------------------------
# Utilidades CLI
# ---------------------------------------------------------------------------

def _buscar_videos_mp4(base_dir: str):
    """Encuentra todos los MP4 originales (excluyendo ya anotados)."""
    base = Path(base_dir)
    return sorted(
        p for p in base.rglob("*.mp4")
        if not p.stem.startswith("web_") and not p.stem.startswith("anotado_")
    )


def _inferir_clase(video_path: Path) -> str:
    """Infiere la clase a partir del directorio padre o el nombre del archivo."""
    clases = {"tecnica_correcta", "codos_abiertos", "rebote_en_pecho", "empuje_desbalanceado"}
    for parte in reversed(video_path.parts[:-1]):
        if parte in clases:
            return parte
    nombre_lower = video_path.stem.lower()
    for c in clases:
        if c in nombre_lower:
            return c
    return "desconocida"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Genera videos de press banca anotados con YOLOv8-pose "
            "y los recodifica a H.264 / yuv420p."
        )
    )
    parser.add_argument(
        "--base-dir", default=".", metavar="DIR",
        help="Directorio raíz con los videos originales (default: .)",
    )
    parser.add_argument(
        "--salida-dir", default="./videos_anotados", metavar="DIR",
        help="Directorio de salida para los videos anotados (default: ./videos_anotados)",
    )
    parser.add_argument(
        "--conf", type=float, default=YOLO_CONF,
        help=f"Umbral de confianza YOLOv8-pose (default: {YOLO_CONF})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Listar videos sin procesarlos.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    videos = _buscar_videos_mp4(args.base_dir)
    if not videos:
        print(f"⚠  No se encontraron videos MP4 en: {args.base_dir}")
        sys.exit(0)

    print(f"Videos encontrados: {len(videos)}")
    for v in videos:
        print(f"  {v}")

    if args.dry_run:
        sys.exit(0)

    if not _verificar_ffmpeg():
        print(
            "\n[AVISO] ffmpeg no detectado en PATH. Los videos se guardarán como AVI.\n"
            "        Instala ffmpeg: https://ffmpeg.org/download.html\n"
        )

    exitosos = fallidos = 0
    for video_path in videos:
        clase = _inferir_clase(video_path)
        print(f"\n>> {video_path.name}  [clase: {clase}]")
        resultado = anotar_video_pose(
            video_path=str(video_path),
            salida_dir=args.salida_dir,
            clase_real=clase,
            pred_rf=clase,
            prob_rf=1.0,
            conf=args.conf,
        )
        if resultado:
            print(f"   OK  -> {resultado}")
            exitosos += 1
        else:
            print("   FALLO")
            fallidos += 1

    print(f"\n{'─' * 50}")
    print(f"Completados: {exitosos}   Fallidos: {fallidos}")


if __name__ == "__main__":
    main()
