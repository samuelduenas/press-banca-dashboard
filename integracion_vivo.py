from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_PATRON_VIVO = re.compile(r"^vivo_(\d{8}_\d{6})\.(csv|mp4)$", re.IGNORECASE)


def listar_pares_capturas_vivo(base_dir: Path) -> List[Dict[str, object]]:
    cap_dir = base_dir / "capturas_vivo"
    if not cap_dir.exists() or not cap_dir.is_dir():
        return []

    por_stamp: Dict[str, Dict[str, Optional[Path]]] = {}
    for item in cap_dir.iterdir():
        if not item.is_file():
            continue
        m = _PATRON_VIVO.match(item.name)
        if not m:
            continue
        stamp, ext = m.group(1), m.group(2).lower()
        entrada = por_stamp.setdefault(stamp, {"csv": None, "mp4": None})
        entrada[ext] = item

    pares: List[Dict[str, object]] = []
    for stamp in sorted(por_stamp.keys(), reverse=True):
        csv_p = por_stamp[stamp]["csv"]
        mp4_p = por_stamp[stamp]["mp4"]
        if csv_p is not None and mp4_p is not None:
            pares.append(
                {
                    "stamp": stamp,
                    "label": f"vivo_{stamp} (CSV + MP4)",
                    "csv_path": csv_p,
                    "mp4_path": mp4_p,
                    "csv_bytes": csv_p.read_bytes(),
                    "mp4_bytes": mp4_p.read_bytes(),
                }
            )
    return pares


def lanzar_en_vivo(
    base_dir: Path,
    port: str = "COM3",
    baud: int = 115200,
    cam: int = 2,
    fs: float = 50.0,
    ventana_seg: float = 4.0,
    proc_actual: Optional[subprocess.Popen] = None,
) -> Tuple[Optional[subprocess.Popen], str]:
    if proc_actual is not None and proc_actual.poll() is None:
        return proc_actual, f"en_vivo.py ya está en ejecución (PID {proc_actual.pid})."

    script_vivo = base_dir / "en_vivo.py"
    if not script_vivo.exists():
        return proc_actual, f"No se encontró {script_vivo}."

    cmd = [
        sys.executable,
        str(script_vivo),
        "--port",
        str(port),
        "--baud",
        str(int(baud)),
        "--cam",
        str(int(cam)),
        "--fs",
        str(float(fs)),
        "--ventana-seg",
        str(float(ventana_seg)),
    ]

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc = subprocess.Popen(
        cmd,
        cwd=str(base_dir),
        creationflags=creationflags,
    )
    return proc, f"Iniciado en_vivo.py (PID {proc.pid}) en {port} @ {baud}, cámara {cam}."


def detener_en_vivo(proc_actual: Optional[subprocess.Popen]) -> Tuple[None, str]:
    if proc_actual is None:
        return None, "No hay ningún proceso en_vivo.py iniciado desde el dashboard."
    if proc_actual.poll() is not None:
        return None, f"en_vivo.py ya había finalizado (código {proc_actual.returncode})."

    proc_actual.terminate()
    try:
        proc_actual.wait(timeout=4.0)
        return None, "Proceso en_vivo.py detenido."
    except subprocess.TimeoutExpired:
        proc_actual.kill()
        proc_actual.wait(timeout=2.0)
        return None, "Proceso en_vivo.py forzado a cerrar."
