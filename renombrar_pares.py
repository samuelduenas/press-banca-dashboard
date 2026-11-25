import os
import subprocess
from pathlib import Path

def obtener_escritorio() -> Path:
    # Obtiene la ruta real del Escritorio en Windows (soporta OneDrive o carpetas redirigidas)
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", '[Environment]::GetFolderPath("Desktop")'],
            text=True
        ).strip()
        if out and Path(out).exists():
            return Path(out)
    except Exception:
        pass
    for candidato in [Path.home() / "Desktop", Path.home() / "OneDrive" / "Desktop", Path.home() / "OneDrive" / "Escritorio"]:
        if candidato.exists():
            return candidato
    return Path.cwd()

DESKTOP_DIR = obtener_escritorio()

# Cambia "press_banca_mpu" si tu carpeta en el Escritorio tiene otro nombre
BASE_DIR = DESKTOP_DIR / "press_banca_mpu"

# Carpeta donde están los videos con subcarpetas por clase adentro (cambia "videos" si se llama distinto)
VIDEOS_DIR = BASE_DIR / "videos"

# True = solo muestra qué renombraría sin cambiar nada | False = renombra los archivos de verdad
SIMULAR = False

CLASES = [
    "tecnica_correcta",
    "codos_abiertos",
    "rebote_en_pecho",
    "empuje_desbalanceado",
]

def recolectar_videos(carpeta_raiz: Path, clase: str):
    if not carpeta_raiz.exists():
        return []
    encontrados = []
    for ext in ("*.mp4", "*.avi", "*.mov"):
        for f in carpeta_raiz.rglob(ext):
            if f.name.startswith("web_"):
                continue
            partes = [p.lower() for p in f.parts]
            if clase.lower() in partes or f.stem.lower().startswith(clase.lower()):
                encontrados.append(f)
    return sorted(set(encontrados))

def renombrar_par(csv_path: Path, vid_path: Path, nuevo_base: str):
    nuevo_csv = csv_path.with_name(f"{nuevo_base}.csv")
    nuevo_vid = vid_path.with_name(f"{nuevo_base}{vid_path.suffix.lower()}")
    print(f"  {csv_path.name}  +  {vid_path.parent.name}\\{vid_path.name}  -->  {nuevo_base}.csv + {nuevo_base}{vid_path.suffix.lower()}")
    if not SIMULAR:
        tmp_csv = csv_path.with_name(f"__tmp_{nuevo_base}.csv")
        tmp_vid = vid_path.with_name(f"__tmp_{nuevo_base}{vid_path.suffix.lower()}")
        csv_path.rename(tmp_csv)
        vid_path.rename(tmp_vid)
        tmp_csv.rename(nuevo_csv)
        tmp_vid.rename(nuevo_vid)

print(f"Escritorio detectado: {DESKTOP_DIR}")
print(f"Carpeta base:         {BASE_DIR}")
print(f"Carpeta videos:       {VIDEOS_DIR}\n")

if not BASE_DIR.exists():
    print("[ERROR] No se encontró la carpeta base. Revisa qué carpetas tienes en tu Escritorio:")
    if DESKTOP_DIR.exists():
        for item in DESKTOP_DIR.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
else:
    # 1. ENTRENAMIENTO (36 pares por clase)
    for clase in CLASES:
        dir_csv = BASE_DIR / clase
        csvs = sorted([f for f in dir_csv.glob("*.csv") if not f.name.startswith("web_")]) if dir_csv.exists() else []
        vids = recolectar_videos(VIDEOS_DIR, clase)
        print(f"[TRAIN] {clase}: {len(csvs)} CSV | {len(vids)} Videos")
        if len(csvs) == len(vids) and len(csvs) > 0:
            for i, (c, v) in enumerate(zip(csvs, vids), start=1):
                renombrar_par(c, v, f"{clase}_{i:03d}")
        else:
            print("  [AVISO] No coincide la cantidad de CSV y videos o revisa la ruta de videos.")

    # 2. PRUEBA (4 pares por clase en pruebas_modelo)
    dir_test = BASE_DIR / "pruebas_modelo"
    if dir_test.exists():
        for clase in CLASES:
            csvs_t = sorted([f for f in dir_test.glob(f"{clase}*.csv") if not f.name.startswith("web_")])
            vids_t = sorted([f for f in list(dir_test.glob(f"{clase}*.mp4")) + list(dir_test.glob(f"{clase}*.avi")) if not f.name.startswith("web_")])
            print(f"\n[TEST] {clase}: {len(csvs_t)} CSV | {len(vids_t)} Videos")
            if len(csvs_t) == len(vids_t) and len(csvs_t) > 0:
                for i, (c, v) in enumerate(zip(csvs_t, vids_t), start=1):
                    renombrar_par(c, v, f"{clase}_test_{i:03d}")
            else:
                print("  [AVISO] No coincide la cantidad de CSV y videos en pruebas_modelo.")
