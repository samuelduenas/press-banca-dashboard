"""
app_dashboard.py — v3.0
=======================
Dashboard Streamlit para visualizar, diagnosticar y validar el clasificador
multimodal (IMU MPU6050 + Pose YOLOv8) de técnica en press de banca.

Modelo        : RandomForest (200 árboles) · 4 clases · features leídas del .joblib
Métricas test : Accuracy 81.25 % (13/16) · F1 macro 0.817   (declaradas por el autor)

Integra (opcionales, mismo directorio):
  · scraper_biomecanica.py → obtener_diagnostico_clase(pred_clase, forzar_actualizar=...)
  · generar_videos_test.py → anotar_video_pose(video_path, salida_dir, clase_real, pred_rf, prob_rf)

Uso:  streamlit run app_dashboard.py
"""

from __future__ import annotations

import hashlib
import html
import importlib.util
import io
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from integracion_vivo import detener_en_vivo, lanzar_en_vivo, listar_pares_capturas_vivo

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("press_banca")

st.set_page_config(
    page_title="Press Banca · Dashboard Multimodal",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# CONSTANTES
# ===========================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

CLASES: List[str] = ["tecnica_correcta", "codos_abiertos", "rebote_en_pecho", "empuje_desbalanceado"]
CLASES_LABEL: Dict[str, str] = {
    "tecnica_correcta": "Técnica Correcta",
    "codos_abiertos": "Codos Abiertos",
    "rebote_en_pecho": "Rebote en Pecho",
    "empuje_desbalanceado": "Empuje Desbalanceado",
}
COLORES_CLASE: Dict[str, str] = {
    "tecnica_correcta": "#14b8a6",
    "codos_abiertos": "#22d3ee",
    "rebote_en_pecho": "#fb7185",
    "empuje_desbalanceado": "#a78bfa",
}

# ---- Métricas declaradas (n=16 test, 4 por clase). NO se recalculan aquí. ----
ACC_TEST, F1_TEST, N_ACIERTOS, N_TEST = 0.8125, 0.817, 13, 16
CLASES_CM = ["codos_abiertos", "empuje_desbalanceado", "rebote_en_pecho", "tecnica_correcta"]
CONF_MATRIX_TEST = np.array(
    [[3, 0, 0, 1],
     [0, 3, 0, 1],
     [0, 0, 4, 0],
     [0, 0, 1, 3]], dtype=int
)  # filas = real, columnas = predicción (orden CLASES_CM)
METRICAS_CLASE = pd.DataFrame(
    [
        ("codos_abiertos", "3/4", 1.000, 0.750, 0.857),
        ("empuje_desbalanceado", "3/4", 1.000, 0.750, 0.857),
        ("rebote_en_pecho", "4/4", 0.800, 1.000, 0.889),
        ("tecnica_correcta", "3/4", 0.600, 0.750, 0.667),
    ],
    columns=["clase", "aciertos", "precision", "recall", "f1"],
)

# ---- Extracción de features desde archivos nuevos (definiciones ASUMIDAS) ----
CANALES_IMU = ["ax", "ay", "az", "gx", "gy", "gz"]
FS_DEFECTO = 50.0
FFT_CORTE_ALTA_HZ = 5.0   # ratio_alta = energía por encima de este corte / energía total (sin DC)
YOLO_CONF = 0.5           # confianza de detección de persona
KP_CONF_MIN = 0.3         # confianza mínima de cada keypoint usado
ALIAS_IMU: Dict[str, set] = {
    "ax": {"ax", "acx", "accx", "accelx", "acelx", "accelerationx"},
    "ay": {"ay", "acy", "accy", "accely", "acely", "accelerationy"},
    "az": {"az", "acz", "accz", "accelz", "acelz", "accelerationz"},
    "gx": {"gx", "gyx", "gyrox", "gyrx", "gyro_x", "gyrox"},
    "gy": {"gy", "gyy", "gyroy", "gyry"},
    "gz": {"gz", "gyz", "gyroz", "gyrz"},
}
ALIAS_TIEMPO = {"time", "tiempo", "timestamp", "t", "ms", "millis", "micros", "ts"}

# ===========================================================================
# TEMAS
# ===========================================================================

TEMAS: Dict[str, Dict[str, str]] = {
    "Oscuro": {
        "bg": "#0b0f19", "bg2": "#111827", "bg3": "#1e2b40", "bg_card": "#162032",
        "text": "#f1f5f9", "text2": "#94a3b8", "text3": "#64748b",
        "accent": "#22d3ee", "accent2": "#14b8a6", "accent3": "#fb7185", "accent4": "#a78bfa",
        "border": "#1e3a5f", "border2": "#2d4a6e",
        "success": "#10b981", "warning": "#f59e0b", "error": "#fb7185",
        "mpl_bg": "#111827", "mpl_bg2": "#1e2b40", "mpl_text": "#e2e8f0", "mpl_grid": "#1e3a5f",
        "mpl_accent": "#22d3ee", "mpl_accent2": "#14b8a6", "mpl_accent3": "#fb7185",
        "mpl_style": "dark_background",
    },
    "Claro": {
        "bg": "#f0f4f8", "bg2": "#ffffff", "bg3": "#e8eef4", "bg_card": "#ffffff",
        "text": "#0f172a", "text2": "#334155", "text3": "#64748b",
        "accent": "#0891b2", "accent2": "#0d9488", "accent3": "#e11d48", "accent4": "#7c3aed",
        "border": "#cbd5e1", "border2": "#94a3b8",
        "success": "#059669", "warning": "#d97706", "error": "#e11d48",
        "mpl_bg": "#ffffff", "mpl_bg2": "#f8fafc", "mpl_text": "#0f172a", "mpl_grid": "#e2e8f0",
        "mpl_accent": "#0891b2", "mpl_accent2": "#0d9488", "mpl_accent3": "#e11d48",
        "mpl_style": "default",
    },
    "Alto Contraste": {
        "bg": "#000000", "bg2": "#0a0a0a", "bg3": "#111111", "bg_card": "#0d0d0d",
        "text": "#ffffff", "text2": "#ffff00", "text3": "#cccccc",
        "accent": "#00ffff", "accent2": "#00ff88", "accent3": "#ff4444", "accent4": "#cc88ff",
        "border": "#444444", "border2": "#666666",
        "success": "#00ff00", "warning": "#ffaa00", "error": "#ff0000",
        "mpl_bg": "#000000", "mpl_bg2": "#111111", "mpl_text": "#ffffff", "mpl_grid": "#333333",
        "mpl_accent": "#00ffff", "mpl_accent2": "#00ff88", "mpl_accent3": "#ff4444",
        "mpl_style": "dark_background",
    },
}


def inyectar_css(t: Dict[str, str]) -> None:
    glow = t["accent"] + "33"
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
    .stApp {{ background-color: {t['bg']}; color: {t['text']}; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {t['bg2']} 0%, {t['bg']} 100%);
        border-right: 1px solid {t['border']};
    }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{ color: {t['text']} !important; }}
    h1 {{ font-size: 1.7rem !important; font-weight: 800 !important; color: {t['text']} !important; }}
    h2 {{ font-size: 1.25rem !important; font-weight: 700 !important; color: {t['text']} !important; }}
    h3, h4 {{ color: {t['text']} !important; font-weight: 600 !important; }}
    p, label, .stMarkdown p {{ color: {t['text2']} !important; }}

    .kpi-wrap {{
        background: {t['bg_card']}; border: 1px solid {t['border']}; border-radius: 16px;
        padding: 22px 18px; text-align: center; position: relative; overflow: hidden;
    }}
    .kpi-wrap::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, {t['accent']}, {t['accent2']});
    }}
    .kpi-icon {{ font-size: 1.5rem; margin-bottom: 6px; display: block; }}
    .kpi-value {{
        font-size: 2.1rem; font-weight: 800; line-height: 1.1; display: block;
        background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }}
    .kpi-label {{ font-size: .72rem; color: {t['text3']}; text-transform: uppercase;
                  letter-spacing: .8px; font-weight: 600; margin-top: 6px; display: block; }}
    .kpi-sub {{ font-size: .75rem; color: {t['text2']}; margin-top: 4px; display: block; }}

    .sec-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 18px;
                   padding-bottom: 12px; border-bottom: 1px solid {t['border']}; }}
    .sec-header-icon {{ width: 36px; height: 36px; border-radius: 10px; font-size: 1.1rem;
        background: linear-gradient(135deg, {t['accent']}22, {t['accent2']}22);
        border: 1px solid {t['accent']}55; display: flex; align-items: center; justify-content: center; }}
    .sec-title {{ font-size: 1.05rem; font-weight: 700; color: {t['text']}; }}

    .badge {{ display: inline-block; padding: 5px 14px; border-radius: 20px; font-size: .82rem;
              font-weight: 600; border: 1px solid; }}
    .pred-panel {{ background: {t['bg3']}; border: 1px solid {t['border']}; border-radius: 12px;
                   padding: 14px 18px; text-align: center; min-height: 86px; }}
    .pred-panel-label {{ font-size: .68rem; color: {t['text3']}; text-transform: uppercase;
                         letter-spacing: .8px; font-weight: 600; margin-bottom: 8px; }}
    .pred-conf-value {{ font-size: 1.8rem; font-weight: 800; line-height: 1; }}

    .fuente-card {{ background: {t['bg3']}; border-left: 3px solid {t['accent']};
                    border-radius: 0 10px 10px 0; padding: 14px 18px; margin-bottom: 12px; }}
    .fuente-titulo {{ font-weight: 600; color: {t['text']}; font-size: .93rem; }}
    .fuente-frag {{ color: {t['text2']}; font-size: .82rem; margin-top: 6px; line-height: 1.55; font-style: italic; }}
    .fuente-url {{ font-size: .75rem; margin-top: 6px; }}
    .fuente-url a {{ color: {t['accent']}; text-decoration: none; }}
    .risk-card {{ background: {t['accent3']}11; border: 1px solid {t['accent3']}44; border-radius: 12px;
                  padding: 18px 22px; margin-bottom: 14px; line-height: 1.65; }}
    .fix-card {{ background: {t['success']}11; border: 1px solid {t['success']}44; border-radius: 12px;
                 padding: 14px 20px; margin-bottom: 12px; }}
    .warn-card {{ background: {t['warning']}22; border: 1px solid {t['warning']}66; border-radius: 12px;
                  padding: 12px 18px; color: {t['warning']}; font-weight: 600; font-size: .86rem;
                  margin-bottom: 16px; }}
    .cache-pill {{ display: inline-block; background: {t['bg3']}; border: 1px solid {t['border']};
                   border-radius: 20px; padding: 3px 11px; font-size: .72rem; color: {t['text3']}; }}

    .stTabs [data-baseweb="tab-list"] {{ background: {t['bg3']}; border-radius: 10px; gap: 2px; padding: 4px; }}
    .stTabs [data-baseweb="tab"] {{ color: {t['text2']}; border-radius: 7px; font-weight: 500; }}
    .stTabs [aria-selected="true"] {{ background: {t['bg_card']} !important; color: {t['accent']} !important;
                                      font-weight: 600; box-shadow: 0 1px 6px {glow}; }}
    .stButton > button {{ background: linear-gradient(135deg, {t['accent']}22, {t['accent2']}22);
        border: 1px solid {t['accent']}66; color: {t['accent']}; border-radius: 10px; font-weight: 600; }}
    .stButton > button:hover {{ border-color: {t['accent']}; box-shadow: 0 4px 16px {glow}; }}
    [data-testid="stFileUploader"] {{ background: {t['bg3']}; border: 1px dashed {t['border2']};
                                      border-radius: 12px; padding: 10px; }}
    hr {{ border-color: {t['border']} !important; opacity: .5; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ===========================================================================
# HELPERS UI (compatibles con versiones viejas y nuevas de Streamlit)
# ===========================================================================

def _imagen(data: bytes) -> None:
    try:
        st.image(data, width="stretch")
    except Exception:
        try:
            st.image(data, use_container_width=True)
        except Exception:
            st.image(data)


def _tabla(df: pd.DataFrame, **kw: Any) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        try:
            st.dataframe(df, use_container_width=True, hide_index=True, **kw)
        except Exception:
            st.dataframe(df)


def fig2st(fig: plt.Figure) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    _imagen(buf.getvalue())


def _sec_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="sec-header"><div class="sec-header-icon">{icon}</div>'
        f'<div class="sec-title">{html.escape(title)}</div></div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# RUTAS / ÍNDICE DE ARCHIVOS
# ===========================================================================

_IGNORAR_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env", ".idea", ".streamlit"}


def detectar_base_dir() -> str:
    for c in [SCRIPT_DIR / "data", SCRIPT_DIR, Path.cwd() / "data", Path.cwd()]:
        if (c / "df_full_fusion.csv").is_file():
            return str(c)
    return str(SCRIPT_DIR)


def buscar_archivo(nombre: str, base_dir: str) -> Optional[str]:
    b = Path(base_dir)
    for c in [b, b / "modelos_guardados", b / "artefactos_modelo", SCRIPT_DIR, Path.cwd()]:
        p = c / nombre
        if p.is_file():
            return str(p.resolve())
    return None


@st.cache_data(show_spinner=False)
def indexar_archivos(base_dir: str, nonce: int = 0) -> Dict[str, List[str]]:
    """Recorre BASE_DIR una sola vez (cacheado). `nonce` permite re-escanear."""
    out: Dict[str, List[str]] = {"video": [], "csv": []}
    for dp, dn, fns in os.walk(base_dir):
        dn[:] = [d for d in dn if d not in _IGNORAR_DIRS]
        for fn in fns:
            low = fn.lower()
            if low.endswith((".mp4", ".mov", ".avi")):
                out["video"].append(os.path.join(dp, fn))
            elif low.endswith(".csv"):
                out["csv"].append(os.path.join(dp, fn))
    return out


def _buscar(paths: List[str], nombres: List[str], excluir_prefijos: Tuple[str, ...] = (),
            excluir_stems: Tuple[str, ...] = ()) -> Optional[str]:
    nombres = [n for n in nombres if n]
    cand = [p for p in paths
            if not Path(p).stem.lower().startswith(excluir_prefijos)
            and Path(p).stem.lower() not in excluir_stems]
    for n in nombres:                      # 1) coincidencia exacta
        for p in cand:
            if Path(p).stem == n:
                return p
    for n in nombres:                      # 2) contiene
        for p in cand:
            if n in Path(p).stem:
                return p
    return None


def buscar_video_original(nombres: List[str], idx: Dict[str, List[str]]) -> Optional[str]:
    return _buscar(idx["video"], nombres, excluir_prefijos=("web_", "anotado_", "_tmp_"))


def buscar_video_anotado(nombres: List[str], idx: Dict[str, List[str]]) -> Optional[str]:
    objetivos = [f"{pre}{n}" for n in nombres if n for pre in ("web_", "anotado_")]
    for p in idx["video"]:
        if Path(p).stem in objetivos:
            return p
    return None


def buscar_csv_imu(nombres: List[str], idx: Dict[str, List[str]]) -> Optional[str]:
    return _buscar(idx["csv"], nombres, excluir_stems=("df_full_fusion",))


# ===========================================================================
# CARGA DE DATOS / MODELO
# ===========================================================================

@st.cache_data(show_spinner=False)
def cargar_df(ruta: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(ruta)


@st.cache_data(show_spinner=False)
def leer_csv_ruta(ruta: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(ruta, sep=None, engine="python")


@st.cache_data(show_spinner=False)
def leer_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), sep=None, engine="python")


@st.cache_resource(show_spinner=False)
def cargar_modelo_raw(ruta: str, mtime: float) -> Any:
    return joblib.load(ruta)


def desempaquetar_modelo(cargado: Any, base_dir: str) -> Tuple[Any, List[str], List[str]]:
    """Devuelve (estimador, features, clases_en_orden_de_predict_proba).

    IMPORTANTE: el orden de predict_proba es `estimador.classes_`, que puede
    diferir de la lista "clases" guardada en el dict del joblib.
    """
    if isinstance(cargado, dict):
        est = cargado.get("modelo")
        feats = [str(f) for f in cargado.get("features", [])]
    else:
        est, feats = cargado, []
    if not feats and hasattr(est, "feature_names_in_"):
        feats = [str(f) for f in est.feature_names_in_]
    if not feats:
        rj = buscar_archivo("feature_cols.json", base_dir)
        if rj:
            try:
                with open(rj, "r", encoding="utf-8") as fh:
                    feats = [str(f) for f in json.load(fh)]
            except Exception:
                feats = []
    clases = [str(c) for c in getattr(est, "classes_", CLASES)]
    return est, feats, clases


def predecir(est: Any, feats: List[str], X_dict: Dict[str, float]) -> Dict[str, Any]:
    """predict + predict_proba alineando EXACTAMENTE nombres y orden del modelo."""
    vals: List[float] = []
    faltantes: List[str] = []
    for f in feats:
        v = X_dict.get(f, np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
        if not np.isfinite(v):
            faltantes.append(f)
            v = 0.0
        vals.append(v)
    arr = np.asarray(vals, dtype=float).reshape(1, -1)
    X = pd.DataFrame(arr, columns=feats) if hasattr(est, "feature_names_in_") else arr
    pred = str(est.predict(X)[0])
    probs = est.predict_proba(X)[0]
    p_dict = {str(c): float(p) for c, p in zip(est.classes_, probs)}   # orden REAL del modelo
    return {
        "pred": pred, "prob_max": p_dict.get(pred, max(p_dict.values())), "probs": p_dict,
        "cubiertas": len(feats) - len(faltantes), "total": len(feats), "faltantes": faltantes,
    }


@st.cache_data(show_spinner=False)
def evaluar_test(ruta_csv: str, mtime_csv: float, ruta_modelo: str, mtime_modelo: float,
                 _est: Any, _feats: List[str]) -> Optional[Dict[str, Any]]:
    """Recalcula accuracy / F1 macro sobre split == 'test' con el modelo cargado."""
    from sklearn.metrics import accuracy_score, f1_score
    df = cargar_df(ruta_csv, mtime_csv)
    if "split" not in df.columns or "clase" not in df.columns:
        return None
    dft = df[df["split"] == "test"]
    if dft.empty:
        return None
    X = dft.reindex(columns=_feats).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X if hasattr(_est, "feature_names_in_") else X.values
    pred = _est.predict(X)
    y = dft["clase"].astype(str).values
    return {
        "n": int(len(dft)), "aciertos": int((pred == y).sum()),
        "acc": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred, average="macro")),
        "cobertura": int(sum(f in df.columns for f in _feats)), "total": len(_feats),
    }


# ===========================================================================
# EXTRACCIÓN DE FEATURES — IMU / FFT  (definiciones estándar ASUMIDAS)
# ===========================================================================

def _norm(c: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def mapear_columnas_imu(df: pd.DataFrame) -> Tuple[Dict[str, str], Optional[str]]:
    """Mapea canales ax..gz a columnas reales (AcX/ax/acc_x...). Coincidencia exacta normalizada."""
    norm = {_norm(c): c for c in df.columns}
    out: Dict[str, str] = {}
    for canal, alias in ALIAS_IMU.items():
        for a in alias:
            if _norm(a) in norm:
                out[canal] = norm[_norm(a)]
                break
        else:   # segundo intento: la columna empieza por el alias (p. ej. 'accxg')
            for a in alias:
                if len(a) >= 3:
                    hit = [orig for k, orig in norm.items() if k.startswith(_norm(a))]
                    if hit:
                        out[canal] = hit[0]
                        break
    col_t = next((norm[k] for k in norm
                  if k in ALIAS_TIEMPO or k.startswith(("time", "tiempo", "millis", "micros"))), None)
    return out, col_t


def estimar_fs(df: pd.DataFrame, col_t: Optional[str]) -> Optional[float]:
    if not col_t:
        return None
    t = pd.to_numeric(df[col_t], errors="coerce").dropna().to_numpy()
    if len(t) < 3:
        return None
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None
    if dt < 0.5:
        return 1.0 / dt            # segundos
    if dt < 500:
        return 1000.0 / dt         # milisegundos
    return 1e6 / dt                # microsegundos


def series_imu(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    mapa, _ = mapear_columnas_imu(df)
    return {c: pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for c, col in mapa.items()}


def _stats_senal(x: np.ndarray, fs: float) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size < 8:
        return {}
    xc = x - x.mean()
    pot = np.abs(np.fft.rfft(xc)) ** 2
    frec = np.fft.rfftfreq(x.size, 1.0 / fs)
    p, f = pot[1:], frec[1:]                       # sin componente DC
    total = float(p.sum()) + 1e-12
    k = int(np.argmax(p))
    return {
        "mean": float(x.mean()), "std": float(x.std()), "rms": float(np.sqrt(np.mean(x ** 2))),
        "rango": float(x.max() - x.min()),
        "fdom": float(f[k]), "ratio_dom": float(p[k] / total),
        "ratio_alta": float(p[f > FFT_CORTE_ALTA_HZ].sum() / total),
    }


def extraer_features_imu(df: pd.DataFrame, fs: float) -> Dict[str, float]:
    sig = series_imu(df)
    cand: Dict[str, np.ndarray] = dict(sig)
    for nombre, trio in (("a_mag", ("ax", "ay", "az")), ("g_mag", ("gx", "gy", "gz"))):
        if all(c in sig for c in trio):
            n = min(len(sig[c]) for c in trio)
            M = np.vstack([sig[c][:n] for c in trio])
            ok = np.isfinite(M).all(axis=0)
            cand[nombre] = np.sqrt((M[:, ok] ** 2).sum(axis=0))
    feats: Dict[str, float] = {}
    for canal, x in cand.items():
        s = _stats_senal(x, fs)
        for k in ("mean", "std", "rms", "rango"):
            if k in s:
                feats[f"imu_{canal}_{k}"] = s[k]
        for k in ("fdom", "ratio_dom", "ratio_alta"):
            if k in s:
                feats[f"fft_{canal}_{k}"] = s[k]
    return feats


# ===========================================================================
# EXTRACCIÓN DE FEATURES — POSE (YOLOv8-pose, definiciones ASUMIDAS)
# ===========================================================================

@st.cache_resource(show_spinner=False)
def cargar_yolo(ruta: str) -> Any:
    from ultralytics import YOLO
    return YOLO(ruta)


def _angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def _agregar(nombre: str, serie: List[float]) -> Dict[str, float]:
    s = np.asarray(serie, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {}
    return {
        f"pose_{nombre}_mean": float(s.mean()), f"pose_{nombre}_std": float(s.std()),
        f"pose_{nombre}_min": float(s.min()), f"pose_{nombre}_max": float(s.max()),
        f"pose_{nombre}_rango": float(s.max() - s.min()),
    }


def extraer_pose_video(ruta_video: str, ruta_yolo: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Persona principal = detección de mayor confianza. Keypoints COCO:
    5/6 hombros, 7/8 codos, 9/10 muñecas (izq/der). Coordenadas en píxeles."""
    meta: Dict[str, Any] = {"frames": 0, "validos": 0, "error": None}
    try:
        import cv2
        modelo = cargar_yolo(ruta_yolo)
    except Exception as exc:
        meta["error"] = f"ultralytics/opencv no disponibles: {exc}"
        return {}, meta
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        meta["error"] = "No se pudo abrir el video."
        return {}, meta
    S: Dict[str, List[float]] = {k: [] for k in
                                 ("ang_codo_izq", "ang_codo_der", "asim_munecas",
                                  "asim_codos", "altura_munecas", "apertura_codos")}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            meta["frames"] += 1
            h = float(frame.shape[0])
            r = modelo(frame, conf=YOLO_CONF, verbose=False)[0]
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                continue
            i = int(np.argmax(r.boxes.conf.cpu().numpy()))
            xy = r.keypoints.xy[i].cpu().numpy()
            kc = (r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None
                  else np.ones(len(xy)))
            if len(xy) < 11 or np.any(kc[5:11] < KP_CONF_MIN):
                continue
            meta["validos"] += 1
            sl, sr, el, er, wl, wr = xy[5], xy[6], xy[7], xy[8], xy[9], xy[10]
            S["ang_codo_izq"].append(_angulo(sl, el, wl))
            S["ang_codo_der"].append(_angulo(sr, er, wr))
            S["asim_munecas"].append(abs(wl[1] - wr[1]) / h)
            S["asim_codos"].append(abs(el[1] - er[1]) / h)
            S["altura_munecas"].append(((sl[1] + sr[1]) / 2 - (wl[1] + wr[1]) / 2) / h)
            S["apertura_codos"].append(abs(el[0] - er[0]) / max(abs(sl[0] - sr[0]), 1e-6))
    finally:
        cap.release()
    feats: Dict[str, float] = {}
    for k, v in S.items():
        feats.update(_agregar(k, v))
    return feats, meta


@st.cache_data(show_spinner=False, max_entries=6)
def pose_cacheada(ruta_video: str, ruta_yolo: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    return extraer_pose_video(ruta_video, ruta_yolo)


def guardar_temp(data: bytes, sufijo: str) -> str:
    h = hashlib.md5(data).hexdigest()
    ruta = Path(tempfile.gettempdir()) / f"pb_{h}{sufijo}"
    if not ruta.exists():
        ruta.write_bytes(data)
    return str(ruta)


# ===========================================================================
# GRÁFICOS
# ===========================================================================

def _safe_style(style: str) -> str:
    return style if style in plt.style.available else "default"


def _ax_style(ax: plt.Axes, t: Dict[str, str]) -> None:
    ax.set_facecolor(t["mpl_bg2"])
    ax.tick_params(colors=t["mpl_text"], labelsize=8.5)
    ax.xaxis.label.set_color(t["mpl_text"])
    ax.yaxis.label.set_color(t["mpl_text"])
    ax.title.set_color(t["mpl_text"])
    for sp in ax.spines.values():
        sp.set_color(t["mpl_grid"])


def _legend(ax: plt.Axes, t: Dict[str, str], **kw: Any) -> None:
    leg = ax.legend(fontsize=9, framealpha=0.2, **kw)
    for txt in leg.get_texts():
        txt.set_color(t["mpl_text"])


def plot_donut(valor: float, label: str, t: Dict[str, str], c1: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    fig.patch.set_facecolor(t["mpl_bg"])
    ax.set_facecolor(t["mpl_bg"])
    ax.pie([valor, max(0.0, 1 - valor)], colors=[c1, t["mpl_bg2"]], startangle=90,
           counterclock=False, wedgeprops={"linewidth": 0, "width": 0.38})
    ax.text(0, 0, f"{valor:.1%}",
            ha="center", va="center", fontsize=14, fontweight="bold", color=c1)
    ax.text(0, -0.42, label, ha="center", va="center", fontsize=7.5, color=t["mpl_text"], alpha=0.8)
    fig.tight_layout(pad=0.1)
    return fig


def plot_confusion(t: Dict[str, str]) -> plt.Figure:
    etiq = ["Codos Abiertos", "Emp. Desbal.", "Rebote", "T. Correcta"]
    base = (["#f8fafc", "#bfdbfe", "#22d3ee", "#0891b2"] if t["mpl_bg"] == "#ffffff"
            else [t["mpl_bg"], t["mpl_bg2"], t["mpl_accent2"], t["mpl_accent"]])
    cmap = LinearSegmentedColormap.from_list("cm", base)
    with plt.style.context(_safe_style(t["mpl_style"])):
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        fig.patch.set_facecolor(t["mpl_bg"])
        _ax_style(ax, t)
        im = ax.imshow(CONF_MATRIX_TEST, cmap=cmap, aspect="auto", vmin=0, vmax=4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(4):
            for j in range(4):
                v = CONF_MATRIX_TEST[i, j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=15, fontweight="bold",
                        color="#0b0f19" if v >= 3 else t["mpl_text"])
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(etiq, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticklabels(etiq, fontsize=8.5)
        ax.set_xlabel("Predicción", fontsize=10); ax.set_ylabel("Real", fontsize=10)
        ax.set_title("Matriz de Confusión — Test (n=16)", fontsize=10.5, fontweight="bold")
        fig.tight_layout()
    return fig


def plot_probabilidades(probs: Dict[str, float], t: Dict[str, str]) -> plt.Figure:
    cls = [c for c in CLASES if c in probs] or list(probs.keys())
    vals = [probs.get(c, 0.0) for c in cls]
    cols = [COLORES_CLASE.get(c, t["mpl_accent"]) for c in cls]
    with plt.style.context(_safe_style(t["mpl_style"])):
        fig, ax = plt.subplots(figsize=(7, 2.9))
        fig.patch.set_facecolor(t["mpl_bg"])
        _ax_style(ax, t)
        bars = ax.barh([CLASES_LABEL.get(c, c) for c in cls], vals, color=cols, alpha=0.85, height=0.52, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(min(v + 0.014, 0.995), bar.get_y() + bar.get_height() / 2, f"{v:.1%}",
                    va="center", ha="left", fontsize=10, color=t["mpl_text"], fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()
        ax.set_xlabel("Probabilidad", fontsize=9)
        ax.xaxis.grid(True, color=t["mpl_grid"], alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_title("Probabilidades por clase (RF)", fontsize=10, fontweight="bold")
        fig.tight_layout()
    return fig


def _fig_vacia(msg: str, t: Dict[str, str], size: Tuple[float, float] = (8, 2.8)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(t["mpl_bg"])
    _ax_style(ax, t)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=t["mpl_text"], fontsize=10, alpha=0.8,
            transform=ax.transAxes)
    return fig


def plot_serie(x: Optional[np.ndarray], fs: float, canal: str, t: Dict[str, str]) -> plt.Figure:
    if x is None:
        return _fig_vacia(f"Sin señal IMU para el canal {canal.upper()}", t)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return _fig_vacia(f"Señal {canal.upper()} demasiado corta", t)
    tiempo = np.arange(len(x)) / fs
    with plt.style.context(_safe_style(t["mpl_style"])):
        fig, ax = plt.subplots(figsize=(8, 2.8))
        fig.patch.set_facecolor(t["mpl_bg"])
        _ax_style(ax, t)
        ax.plot(tiempo, x, color=t["mpl_accent"], linewidth=1.3, alpha=0.92)
        ax.fill_between(tiempo, x, float(x.mean()), alpha=0.13, color=t["mpl_accent"])
        ax.set_xlabel("Tiempo (s)", fontsize=9); ax.set_ylabel(f"{canal.upper()} (u.a.)", fontsize=9)
        ax.set_title(f"Serie temporal — {canal.upper()} (fs = {fs:.1f} Hz)", fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, color=t["mpl_grid"], alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        fig.tight_layout()
    return fig


def plot_fft(x: Optional[np.ndarray], fs: float, canal: str, t: Dict[str, str]) -> plt.Figure:
    if x is None:
        return _fig_vacia(f"Sin señal IMU para el canal {canal.upper()}", t)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return _fig_vacia(f"Señal {canal.upper()} demasiado corta", t)
    mag = np.abs(np.fft.rfft(x - x.mean())) / x.size
    xf = np.fft.rfftfreq(x.size, 1.0 / fs)
    k = int(np.argmax(mag[1:])) + 1                     # sin DC
    with plt.style.context(_safe_style(t["mpl_style"])):
        fig, ax = plt.subplots(figsize=(8, 2.8))
        fig.patch.set_facecolor(t["mpl_bg"])
        _ax_style(ax, t)
        ax.plot(xf, mag, color=t["mpl_accent2"], linewidth=1.3, alpha=0.92)
        ax.fill_between(xf, mag, alpha=0.13, color=t["mpl_accent2"])
        ax.axvline(xf[k], color=t["mpl_accent3"], linewidth=1.6, linestyle="--", label=f"F. dom.: {xf[k]:.2f} Hz")
        ax.set_xlim(0, fs / 2)
        ax.set_xlabel("Frecuencia (Hz)", fontsize=9); ax.set_ylabel("Magnitud", fontsize=9)
        ax.set_title(f"Espectro FFT (sin DC) — {canal.upper()}", fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, color=t["mpl_grid"], alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        _legend(ax, t)
        fig.tight_layout()
    return fig


def plot_importancia(est: Optional[Any], feats: List[str], t: Dict[str, str]) -> plt.Figure:
    if est is None or not feats or not hasattr(est, "feature_importances_"):
        return _fig_vacia("Modelo no cargado: no hay importancias", t, (9, 3))
    imp = np.asarray(est.feature_importances_)
    top = np.argsort(imp)[::-1][:10]
    nombres = [feats[i] for i in top][::-1]
    vals = [float(imp[i]) for i in top][::-1]
    cols = [t["mpl_accent3"] if n.startswith("pose_") else t["mpl_accent"] for n in nombres]
    with plt.style.context(_safe_style(t["mpl_style"])):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor(t["mpl_bg"])
        _ax_style(ax, t)
        bars = ax.barh(nombres, vals, color=cols, alpha=0.86, height=0.6, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.0007, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center",
                    ha="left", fontsize=8.5, color=t["mpl_text"])
        ax.set_xlim(0, max(vals) * 1.15)
        _legend(ax, t, handles=[Patch(facecolor=t["mpl_accent"], label="IMU / FFT"),
                                Patch(facecolor=t["mpl_accent3"], label="Pose (YOLOv8)")], loc="lower right")
        ax.set_xlabel("Importancia (Gini)", fontsize=9)
        ax.set_title(f"Top 10 features — RF ({len(feats)} variables)", fontsize=10, fontweight="bold")
        ax.xaxis.grid(True, color=t["mpl_grid"], alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        fig.tight_layout()
    return fig


# ===========================================================================
# MÓDULOS AUXILIARES DEL PROYECTO
# ===========================================================================

@st.cache_resource(show_spinner=False)
def _importar_modulo(nombre: str) -> Optional[Any]:
    ruta = SCRIPT_DIR / f"{nombre}.py"
    if not ruta.exists():
        return None
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        logger.warning("Error al importar %s: %s", nombre, exc)
        return None


# ===========================================================================
# SIDEBAR
# ===========================================================================

def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:14px 0 6px"><div style="font-size:2rem">🏋️</div>'
            '<div style="font-size:1.1rem;font-weight:800;background:linear-gradient(135deg,#22d3ee,#14b8a6);'
            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">PRESS BANCA</div>'
            '<div style="font-size:.68rem;color:#64748b;letter-spacing:1px;text-transform:uppercase">'
            'Dashboard Multimodal v3.0</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        base_dir = st.text_input("📁 BASE_DIR", value=detectar_base_dir(), key="base_dir",
                                 help="Carpeta raíz con df_full_fusion.csv, videos y CSV IMU.")
        tema_nombre = st.selectbox("🎨 Tema", list(TEMAS.keys()), index=0, key="tema_sel")

        st.markdown("**Señal IMU (archivos subidos)**")
        fs_manual = st.number_input("Frecuencia de muestreo (Hz)", min_value=1.0, max_value=2000.0,
                                    value=FS_DEFECTO, step=1.0, key="fs_manual")
        fs_auto = st.checkbox("Estimar fs desde la columna de tiempo (si existe)", value=False, key="fs_auto")

        st.markdown("---")
        forzar = st.button("🔄 Forzar actualización scraper", key="btn_scraper")
        rescan = st.button("📂 Re-escanear carpetas", key="btn_rescan")
        if rescan:
            st.session_state["scan_nonce"] = st.session_state.get("scan_nonce", 0) + 1
        st.markdown("---")
        st.markdown("**Captura en vivo (`en_vivo.py`)**")
        if "vivo_proc" not in st.session_state:
            st.session_state["vivo_proc"] = None

        c_p, c_b, c_c = st.columns(3)
        with c_p:
            vivo_port = st.text_input("Puerto", value="COM3", key="vivo_port")
        with c_b:
            vivo_baud = st.number_input("Baudios", value=115200, step=1200, key="vivo_baud")
        with c_c:
            vivo_cam = st.number_input("Cámara", value=2, min_value=0, step=1, key="vivo_cam")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("▶️ Iniciar", key="btn_vivo_start", width="stretch"):
                proc, msg = lanzar_en_vivo(
                    Path(base_dir),
                    port=vivo_port,
                    baud=int(vivo_baud),
                    cam=int(vivo_cam),
                    proc_actual=st.session_state["vivo_proc"],
                )
                st.session_state["vivo_proc"] = proc
                st.caption(msg)
        with b2:
            if st.button("⏹️ Detener", key="btn_vivo_stop", width="stretch"):
                proc, msg = detener_en_vivo(st.session_state["vivo_proc"])
                st.session_state["vivo_proc"] = proc
                st.caption(msg)

        pares_vivo = listar_pares_capturas_vivo(Path(base_dir))
        if pares_vivo:
            mapa_pares = {p["label"]: p for p in pares_vivo}
            sel_vivo = st.selectbox("Par en capturas_vivo/", list(mapa_pares.keys()), key="sel_vivo_par")
            if st.button("📥 Cargar par vivo", key="btn_cargar_vivo_par", width="stretch"):
                st.session_state["par_vivo_activo"] = mapa_pares[sel_vivo]
                st.session_state["vivo_nonce"] = st.session_state.get("vivo_nonce", 0) + 1
                st.session_state["fuente_activa"] = "Par subido"
                st.caption(f"Cargado: {sel_vivo}")
        ruta_csv = buscar_archivo("df_full_fusion.csv", base_dir)
        ruta_modelo = buscar_archivo("modelo_press_banca_rf.joblib", base_dir)
        ruta_yolo = buscar_archivo("yolov8n-pose.pt", base_dir)
        st.markdown("---")
        for ok, lbl in ((ruta_csv, "Dataset"), (ruta_modelo, "Modelo RF"), (ruta_yolo, "YOLOv8n-pose")):
            st.markdown(f"{'🟢' if ok else '🔴'} **{lbl}** {'OK' if ok else 'no encontrado'}")
        st.caption("RF 200 árboles · 4 clases · IMU + Pose")

    return {
        "base_dir": base_dir, "tema": TEMAS[tema_nombre], "tema_nombre": tema_nombre,
        "fs_manual": float(fs_manual), "fs_auto": bool(fs_auto), "forzar": bool(forzar),
        "ruta_csv": ruta_csv, "ruta_modelo": ruta_modelo, "ruta_yolo": ruta_yolo or "yolov8n-pose.pt",
    }


# ===========================================================================
# SECCIÓN 1 — KPIs
# ===========================================================================

def render_kpis(cfg: Dict[str, Any], est: Optional[Any], feats: List[str]) -> None:
    t = cfg["tema"]
    _sec_header("📈", "KPIs y Métricas en Test")
    n_imu = sum(not f.startswith("pose_") for f in feats)
    n_pose = sum(f.startswith("pose_") for f in feats)
    kpis = [
        ("🎯", f"{ACC_TEST:.2%}", "Accuracy Test", f"{N_ACIERTOS} / {N_TEST} correctas"),
        ("📊", f"{F1_TEST:.3f}", "F1 Macro (Test)", "precisión 0.850 · recall 0.812"),
        ("📦", "160", "Total de pares", "144 train · 16 test"),
        ("🔬", str(len(feats)) if feats else "—", "Features del modelo",
         f"{n_imu} IMU/FFT + {n_pose} pose" if feats else "modelo no cargado"),
    ]
    for col, (icon, val, label, sub) in zip(st.columns(4), kpis):
        with col:
            st.markdown(
                f'<div class="kpi-wrap"><span class="kpi-icon">{icon}</span><span class="kpi-value">{val}</span>'
                f'<span class="kpi-label">{label}</span><span class="kpi-sub">{sub}</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2.6])
    with c1:
        fig2st(plot_donut(ACC_TEST, "Accuracy Test", t, t["mpl_accent"]))
    with c2:
        fig2st(plot_donut(F1_TEST, "F1 Macro Test", t, t["mpl_accent2"]))
    with c3:
        st.markdown("**Desglose por clase (test, n = 16)**")
        dfm = METRICAS_CLASE.copy()
        dfm["clase"] = dfm["clase"].map(CLASES_LABEL)
        _tabla(dfm)
        st.caption("Métricas declaradas por el autor; no son un recálculo de esta app.")

    with st.expander("🔍 Matriz de confusión y verificación en vivo con el modelo cargado", expanded=False):
        cm_col, ver_col = st.columns([1, 1.2])
        with cm_col:
            fig2st(plot_confusion(t))
        with ver_col:
            st.markdown("**Verificación en vivo (split = test)**")
            res = None
            if est is not None and feats and cfg["ruta_csv"] and cfg["ruta_modelo"]:
                try:
                    res = evaluar_test(cfg["ruta_csv"], os.path.getmtime(cfg["ruta_csv"]),
                                       cfg["ruta_modelo"], os.path.getmtime(cfg["ruta_modelo"]), est, feats)
                except Exception as exc:
                    st.warning(f"No se pudo evaluar: {exc}")
            if res is None:
                st.caption("Requiere dataset con columnas `split` y `clase` y el modelo cargado.")
            else:
                st.markdown(
                    f"- Muestras test: **{res['n']}** · aciertos **{res['aciertos']}** · "
                    f"accuracy **{res['acc']:.2%}** · F1 macro **{res['f1']:.3f}**\n"
                    f"- Features del modelo presentes en el CSV: **{res['cobertura']}/{res['total']}**"
                )
                if res["cobertura"] < res["total"]:
                    st.warning("El CSV no contiene todas las features del modelo: las faltantes se rellenan "
                               "con 0.0 y las cifras de arriba no son comparables con las declaradas.")
                elif abs(res["acc"] - ACC_TEST) > 1e-6:
                    st.warning("La accuracy recalculada difiere de la declarada (81.25 %). "
                               "Revisa que modelo y dataset sean de la misma corrida.")


# ===========================================================================
# SECCIÓN 2 — EXPLORADOR
# ===========================================================================

def _render_cobertura(res: Dict[str, Any]) -> None:
    n, tot = res["cubiertas"], res["total"]
    txt = f"Cobertura de features: **{n}/{tot}** reales · **{tot - n}** rellenadas con 0.0"
    if tot - n:
        faltan = ", ".join(f"`{f}`" for f in res["faltantes"][:12])
        txt += f" — faltan: {faltan}{' …' if tot - n > 12 else ''}"
    st.markdown(txt)


def _render_pred_badges(clase_real: str, res: Dict[str, Any], t: Dict[str, str], hide_real: bool = False) -> None:
    pred, prob = res["pred"], res["prob_max"]
    c_real = COLORES_CLASE.get(clase_real, t["accent"])
    c_pred = COLORES_CLASE.get(pred, t["accent"])
    acierto = (not hide_real) and clase_real == pred
    c_conf = t["success"] if (acierto or hide_real) else t["error"]
    icono = "" if hide_real else ("✅" if acierto else "❌")
    b1, b2, b3 = st.columns(3)
    with b1:
        inner = ('<span style="color:%s;font-size:.85rem">— (muestra nueva)</span>' % t["text3"] if hide_real else
                 f'<span class="badge" style="color:{c_real};border-color:{c_real};background:{c_real}18">'
                 f'{CLASES_LABEL.get(clase_real, clase_real)}</span>')
        st.markdown(f'<div class="pred-panel"><div class="pred-panel-label">CLASE REAL</div>{inner}</div>',
                    unsafe_allow_html=True)
    with b2:
        st.markdown(
            f'<div class="pred-panel"><div class="pred-panel-label">PREDICCIÓN RF</div>'
            f'<span class="badge" style="color:{c_pred};border-color:{c_pred};background:{c_pred}18">'
            f'{CLASES_LABEL.get(pred, pred)}</span></div>', unsafe_allow_html=True)
    with b3:
        st.markdown(
            f'<div class="pred-panel"><div class="pred-panel-label">CONFIANZA {icono}</div>'
            f'<div class="pred-conf-value" style="color:{c_conf}">{prob:.1%}</div></div>',
            unsafe_allow_html=True)


def _fs_para(df_imu: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[float, str]:
    if cfg["fs_auto"]:
        _, col_t = mapear_columnas_imu(df_imu)
        fs = estimar_fs(df_imu, col_t)
        if fs and 1.0 <= fs <= 2000.0:
            return fs, "estimada desde la columna de tiempo"
    return cfg["fs_manual"], "manual (sidebar)"


def _panel_dataset(df: pd.DataFrame, est: Optional[Any], feats: List[str],
                   idx: Dict[str, List[str]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = cfg["tema"]
    splits = [s for s in ("test", "train") if "split" in df.columns and (df["split"] == s).any()]
    opciones = splits + ["todos"]
    c1, c2 = st.columns([1, 2])
    with c1:
        split_sel = st.selectbox("Split", opciones, index=0, key="split_ds")
    df_f = df if split_sel == "todos" or "split" not in df.columns else df[df["split"] == split_sel]
    if df_f.empty:
        st.warning("El dataset está vacío para ese split.")
        return {}
    col_nombre = next((c for c in ("archivo_base", "archivo_base_pose") if c in df_f.columns), df_f.columns[0])
    tomas = df_f[col_nombre].astype(str).tolist()
    with c2:
        toma = st.selectbox("Toma", tomas, key="toma_ds")
    row = df_f[df_f[col_nombre].astype(str) == toma].iloc[0]
    clase_real = str(row.get("clase", "desconocida"))

    if est is None or not feats:
        st.warning("Modelo no cargado: no se puede predecir.")
        return {}
    X = {f: pd.to_numeric(row.get(f), errors="coerce") for f in feats if f in row.index}
    res = predecir(est, feats, X)

    nombres = [toma] + [str(row[c]) for c in ("archivo_base", "archivo_base_pose", "archivo_base_imu")
                        if c in row.index and str(row[c]) != toma]
    csv_path = buscar_csv_imu(nombres, idx)
    series: Dict[str, np.ndarray] = {}
    fs, fs_origen = cfg["fs_manual"], "manual (sidebar)"
    if csv_path:
        try:
            df_imu = leer_csv_ruta(csv_path, os.path.getmtime(csv_path))
            series = series_imu(df_imu)
            fs, fs_origen = _fs_para(df_imu, cfg)
        except Exception as exc:
            logger.warning("CSV IMU ilegible: %s", exc)

    _render_pred_badges(clase_real, res, t)
    st.markdown("<br>", unsafe_allow_html=True)
    _render_cobertura(res)
    fig2st(plot_probabilidades(res["probs"], t))
    return {
        "modo": "dataset", "nombre": toma, "nombres": nombres, "clase_real": clase_real, "res": res,
        "row": row, "series": series, "fs": fs, "fs_origen": fs_origen, "csv_path": csv_path,
        "video_tmp": None,
    }


def _panel_subida(est: Optional[Any], feats: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = cfg["tema"]
    st.markdown("Sube un CSV del MPU6050 y/o un video de la ejecución para clasificar una toma nueva.")
    c1, c2 = st.columns(2)
    with c1:
        csv_file = st.file_uploader("📄 CSV del MPU6050", type=["csv"], key="up_csv")
    with c2:
        vid_file = st.file_uploader("🎬 Video", type=["mp4", "mov", "avi"], key="up_vid")

        par_vivo = st.session_state.get("par_vivo_activo")
        vivo_stamp = par_vivo.get("stamp") if isinstance(par_vivo, dict) else None
        es_par_vivo_nuevo = (
            vivo_stamp is not None
            and st.session_state.get("vivo_stamp_visto") != vivo_stamp
        )

        if es_par_vivo_nuevo and par_vivo is not None:
            csv_bytes = par_vivo["csv_bytes"]
            csv_name = f"vivo_{vivo_stamp}.csv"
            vid_bytes = par_vivo["mp4_bytes"]
            vid_name = f"vivo_{vivo_stamp}.mp4"
            st.session_state["vivo_stamp_visto"] = vivo_stamp
            st.caption(f"Usando captura guardada: `{par_vivo['label']}`")
        else:
            csv_bytes = csv_file.getvalue() if csv_file is not None else None
            csv_name = csv_file.name if csv_file is not None else None
            vid_bytes = vid_file.getvalue() if vid_file is not None else None
            vid_name = vid_file.name if vid_file is not None else None
            if csv_bytes is None and vid_bytes is None and par_vivo is not None:
                csv_bytes = par_vivo["csv_bytes"]
                csv_name = f"vivo_{par_vivo['stamp']}.csv"
                vid_bytes = par_vivo["mp4_bytes"]
                vid_name = f"vivo_{par_vivo['stamp']}.mp4"
                st.session_state["vivo_stamp_visto"] = vivo_stamp
                st.caption(f"Usando captura guardada: `{par_vivo['label']}`")
    if csv_bytes is None and vid_bytes is None:
        st.caption("Sube al menos un archivo o carga un par desde capturas_vivo/.")
        return {}
    if est is None or not feats:
        st.warning("Modelo no cargado: no se puede predecir.")
        return {}

    X: Dict[str, float] = {}
    series: Dict[str, np.ndarray] = {}
    fs, fs_origen = cfg["fs_manual"], "manual (sidebar)"
    video_tmp: Optional[str] = None
    msgs: List[str] = []

    if csv_bytes is not None:
        try:
            df_up = leer_csv_bytes(csv_bytes)
            mapa, _ = mapear_columnas_imu(df_up)
            fs, fs_origen = _fs_para(df_up, cfg)
            series = series_imu(df_up)
            f_imu = extraer_features_imu(df_up, fs)
            X.update(f_imu)
            msgs.append(f"CSV: canales detectados {', '.join(mapa) or 'ninguno'} · {len(f_imu)} features IMU/FFT "
                        f"· fs = {fs:.1f} Hz ({fs_origen})")
        except Exception as exc:
            st.error(f"No se pudo leer el CSV: {exc}")

    if vid_bytes is not None:
        try:
            video_tmp = guardar_temp(vid_bytes, Path(vid_name or "video.mp4").suffix or ".mp4")
            with st.spinner("Extrayendo pose con YOLOv8-pose (solo la primera vez)…"):
                f_pose, meta = pose_cacheada(video_tmp, cfg["ruta_yolo"])
            if meta.get("error"):
                msgs.append(f"Pose: {meta['error']}")
            else:
                X.update(f_pose)
                msgs.append(f"Pose: {len(f_pose)} features · {meta['validos']}/{meta['frames']} frames válidos")
        except Exception as exc:
            st.error(f"Error procesando el video: {exc}")

    for m in msgs:
        st.caption(m)
    if not X:
        st.warning("No se extrajeron features.")
        return {}

    res = predecir(est, feats, X)
    _render_pred_badges("desconocida", res, t, hide_real=True)
    st.markdown("<br>", unsafe_allow_html=True)
    _render_cobertura(res)
    if res["cubiertas"] < res["total"]:
        st.warning("Solo subiste una modalidad (o faltan columnas): las features no cubiertas valen 0.0, "
                   "así que la predicción es poco fiable.")
    st.caption("⚠️ Las fórmulas de extracción de archivos nuevos son definiciones estándar asumidas; "
               "valídalas con la sección «Verificar extracción» antes de confiar en el resultado.")
    fig2st(plot_probabilidades(res["probs"], t))
    nombre = vid_name or csv_name or "captura_nueva"
    return {
        "modo": "subida", "nombre": Path(nombre).stem, "nombres": [Path(nombre).stem],
        "clase_real": "desconocida", "res": res, "row": pd.Series(X, dtype=float),
        "series": series, "fs": fs, "fs_origen": fs_origen, "csv_path": None, "video_tmp": video_tmp,
    }


def render_explorador(df: pd.DataFrame, est: Optional[Any], feats: List[str],
                      idx: Dict[str, List[str]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    _sec_header("🔎", "Explorador Interactivo de Tomas")
    tab_ds, tab_up = st.tabs(["📊  Muestras del dataset", "📤  Subir par nuevo (CSV + Video)"])
    with tab_ds:
        info_ds = _panel_dataset(df, est, feats, idx, cfg) if not df.empty else {}
        if df.empty:
            st.warning("No se encontró df_full_fusion.csv. Ajusta BASE_DIR en el sidebar.")
    with tab_up:
        info_up = _panel_subida(est, feats, cfg)
    if "fuente_activa" not in st.session_state:
        st.session_state["fuente_activa"] = "Muestra del dataset"
    
    fuente = st.radio("Fuente para las secciones siguientes", ["Muestra del dataset", "Par subido"],
                      horizontal=True, key="fuente_activa")
    if fuente == "Par subido":
        st.caption("" if info_up else "Aún no hay par subido: se muestra la muestra del dataset.")
        return info_up or info_ds
    st.caption("")
    return info_ds


# ===========================================================================
# SECCIÓN 3 — VISIÓN (video)
# ===========================================================================

def render_vision(info: Dict[str, Any], cfg: Dict[str, Any], idx: Dict[str, List[str]]) -> None:
    _sec_header("🎥", "Visión Artificial (YOLOv8-pose) · Video")
    if not info:
        st.caption("Selecciona una toma en el explorador.")
        return
    res = info["res"]
    nombres = info.get("nombres", [info["nombre"]])
    clave = f"vid_anot::{info['nombre']}"

    if info["modo"] == "subida":
        video_orig = info.get("video_tmp")
        video_anot = st.session_state.get(clave)
    else:
        video_orig = buscar_video_original(nombres, idx)
        video_anot = st.session_state.get(clave) or buscar_video_anotado(nombres, idx)
    if video_anot and not os.path.isfile(video_anot):
        video_anot = None

    col_vid, col_ctrl = st.columns([2, 1])

    with col_ctrl:                              # se procesa primero para refrescar el video sin st.rerun()
        st.markdown("**Información**")
        st.markdown(f"- **Clase real:** {CLASES_LABEL.get(info['clase_real'], info['clase_real'])}")
        st.markdown(f"- **Pred RF:** {CLASES_LABEL.get(res['pred'], res['pred'])}")
        st.markdown(f"- **Confianza:** `{res['prob_max']:.1%}`")
        gen = st.button("🤖 Generar / regenerar video anotado", key="btn_gen_vid")
        msg = ""
        if gen:
            gvt = _importar_modulo("generar_videos_test")
            if gvt is None:
                msg = "`generar_videos_test.py` no encontrado junto a app_dashboard.py."
            elif not video_orig or not os.path.isfile(video_orig):
                msg = "No hay video original para anotar."
            else:
                try:
                    with st.spinner("Aplicando YOLOv8-pose…"):
                        salida = gvt.anotar_video_pose(
                            video_path=video_orig, salida_dir=str(Path(video_orig).parent),
                            clase_real=info["clase_real"], pred_rf=res["pred"], prob_rf=res["prob_max"])
                    if salida:
                        st.session_state[clave] = salida
                        video_anot = salida
                        msg = f"Guardado: {Path(salida).name}"
                    else:
                        msg = "No se pudo generar el video."
                except Exception as exc:
                    msg = f"Error: {exc}"
        st.caption(msg)

    with col_vid:
        mostrar = video_anot or video_orig
        if mostrar and os.path.isfile(mostrar):
            etiqueta = "anotado" if mostrar == video_anot else "original (sin anotar)"
            st.markdown(f"**Video {etiqueta}:** `{Path(mostrar).name}`")
            try:
                st.video(mostrar)
            except Exception as exc:
                st.warning(f"No se pudo reproducir: {exc}")
        else:
            st.markdown("**Video:** no encontrado")
            st.caption(f"No hay video para `{info['nombre']}` en BASE_DIR.")


# ===========================================================================
# SECCIÓN 4 — IMU + FFT + IMPORTANCIAS + VERIFICACIÓN
# ===========================================================================

def _comparar(row: pd.Series, extraidas: Dict[str, float]) -> pd.DataFrame:
    filas = []
    for f, v in extraidas.items():
        if f in row.index and pd.notna(row[f]):
            d = float(row[f])
            err = abs(v - d) / max(abs(d), 1e-9)
            filas.append((f, d, v, err))
    out = pd.DataFrame(filas, columns=["feature", "dataset", "extraida", "err_rel"])
    return out.sort_values("err_rel", ascending=False) if not out.empty else out


def render_imu_fft(info: Dict[str, Any], est: Optional[Any], feats: List[str], cfg: Dict[str, Any]) -> None:
    t = cfg["tema"]
    _sec_header("📡", "Señal IMU (MPU6050) + FFT + Importancia de Variables")
    if not info:
        st.caption("Selecciona una toma en el explorador.")
        return

    canal = st.radio("Canal IMU", CANALES_IMU, index=2, horizontal=True, key="canal_imu")
    series, fs = info.get("series", {}), float(info.get("fs", cfg["fs_manual"]))
    x = series.get(canal)
    if info["modo"] == "dataset":
        st.caption(f"CSV: `{Path(info['csv_path']).name}` · fs = {fs:.1f} Hz ({info['fs_origen']})"
                   if info.get("csv_path") else "No se encontró un CSV IMU para esta toma en BASE_DIR.")
    else:
        st.caption(f"Señal del CSV subido · fs = {fs:.1f} Hz ({info['fs_origen']})" if series
                   else "No se subió CSV IMU.")
    p1, p2 = st.columns(2)
    with p1:
        fig2st(plot_serie(x, fs, canal, t))
    with p2:
        fig2st(plot_fft(x, fs, canal, t))

    st.markdown("---")
    st.markdown("**🌟 Top 10 features — Random Forest**")
    fig2st(plot_importancia(est, feats, t))

    with st.expander("🧪 Verificar extracción vs dataset (solo muestras del dataset)", expanded=False):
        st.caption("Compara las features que esta app calcula desde el CSV/video crudo con las guardadas en "
                   "df_full_fusion.csv para la misma toma. Si el error es grande, la definición asumida no "
                   "coincide con la usada al entrenar.")
        if info["modo"] != "dataset" or not info.get("csv_path"):
            st.caption("Disponible cuando la toma es del dataset y su CSV IMU existe en BASE_DIR.")
        else:
            try:
                df_imu = leer_csv_ruta(info["csv_path"], os.path.getmtime(info["csv_path"]))
                comp = _comparar(info["row"], extraer_features_imu(df_imu, fs))
                if comp.empty:
                    st.caption("Sin features comparables (¿nombres distintos en el dataset?).")
                else:
                    ok = int((comp["err_rel"] <= 0.05).sum())
                    st.markdown(f"IMU/FFT: **{ok}/{len(comp)}** features coinciden dentro de 5 % de error relativo.")
                    _tabla(comp.head(15))
            except Exception as exc:
                st.warning(f"No se pudo comparar IMU: {exc}")
        run_pose = st.button("Comparar también pose (lento: corre YOLO sobre el video original)", key="btn_val_pose")
        if run_pose and info["modo"] == "dataset":
            idx = indexar_archivos(cfg["base_dir"], st.session_state.get("scan_nonce", 0))
            v = buscar_video_original(info.get("nombres", []), idx)
            if not v:
                st.caption("No se encontró el video original de esta toma.")
            else:
                with st.spinner("Procesando video…"):
                    f_pose, meta = pose_cacheada(v, cfg["ruta_yolo"])
                comp = _comparar(info["row"], f_pose)
                if meta.get("error") or comp.empty:
                    st.warning(meta.get("error") or "Sin features de pose comparables.")
                else:
                    ok = int((comp["err_rel"] <= 0.05).sum())
                    st.markdown(f"Pose: **{ok}/{len(comp)}** features coinciden dentro de 5 %.")
                    _tabla(comp.head(15))


# ===========================================================================
# SECCIÓN 5 — DIAGNÓSTICO BIOMECÁNICO
# ===========================================================================

def render_diagnostico(info: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    t = cfg["tema"]
    _sec_header("🩺", "Diagnóstico y Corrección Biomecánica")
    if not info:
        st.caption("Selecciona una toma en el explorador.")
        return
    pred = info["res"]["pred"]
    scraper = _importar_modulo("scraper_biomecanica")
    if scraper is None:
        st.warning("`scraper_biomecanica.py` no disponible junto a app_dashboard.py.")
        return

    clave = f"diag::{pred}"
    diag = st.session_state.get(clave)
    if diag is None or cfg["forzar"]:
        try:
            with st.spinner(f"Obteniendo diagnóstico para {CLASES_LABEL.get(pred, pred)}…"):
                diag = scraper.obtener_diagnostico_clase(pred, forzar_actualizar=cfg["forzar"])
            st.session_state[clave] = diag
        except Exception as exc:
            st.error(f"Error en scraper: {exc}")
            return
    if not diag:
        st.warning("El scraper no devolvió datos.")
        return

    color = COLORES_CLASE.get(pred, t["accent"])
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f'<h3>Clase: <span style="color:{color}">{CLASES_LABEL.get(pred, pred)}</span></h3>',
                    unsafe_allow_html=True)
    with h2:
        pill = "💾 Caché" if diag.get("desde_cache") else "🌐 En vivo"
        st.markdown(f'<div style="text-align:right;margin-top:16px"><span class="cache-pill">{pill}</span></div>',
                    unsafe_allow_html=True)

    riesgo = html.escape(str(diag.get("riesgo", "Sin información.")))
    st.markdown(
        f'<div class="risk-card"><strong style="color:{t["error"]}">⚠️ Riesgo articular / muscular</strong>'
        f'<br><br><span style="color:{t["text2"]}">{riesgo}</span></div>', unsafe_allow_html=True)

    corr = diag.get("correccion", []) or []
    st.markdown(f'<div class="fix-card"><strong style="color:{t["success"]}">✅ Cómo corregirlo</strong></div>',
                unsafe_allow_html=True)
    if corr:
        st.markdown("\n".join(f"{i}. {p}" for i, p in enumerate(corr, 1)))
    else:
        st.caption("Sin pasos de corrección disponibles.")

    fuentes = diag.get("fuentes", []) or []
    st.markdown(f"**📚 Fuentes biomecánicas ({len(fuentes)})**")
    for f in fuentes:
        url = str(f.get("url", "#"))
        safe_url = html.escape(url, quote=True) if url.startswith(("http://", "https://")) else "#"
        st.markdown(
            f'<div class="fuente-card"><div class="fuente-titulo">📖 {html.escape(str(f.get("titulo", "Sin título")))}</div>'
            f'<div class="fuente-frag">"{html.escape(str(f.get("fragmento", "")))}"</div>'
            f'<div class="fuente-url">🔗 <a href="{safe_url}" target="_blank" rel="noopener">{html.escape(url)}</a></div></div>',
            unsafe_allow_html=True)
    if not fuentes:
        st.caption("No se encontraron fuentes en esta consulta.")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    cfg = render_sidebar()
    inyectar_css(cfg["tema"])
    t = cfg["tema"]
    idx = indexar_archivos(cfg["base_dir"], st.session_state.get("scan_nonce", 0))

    df = pd.DataFrame()
    if cfg["ruta_csv"]:
        try:
            df = cargar_df(cfg["ruta_csv"], os.path.getmtime(cfg["ruta_csv"]))
        except Exception as exc:
            st.error(f"Error al cargar el dataset: {exc}")

    est: Optional[Any] = None
    feats: List[str] = []
    if cfg["ruta_modelo"]:
        try:
            raw = cargar_modelo_raw(cfg["ruta_modelo"], os.path.getmtime(cfg["ruta_modelo"]))
            est, feats, _ = desempaquetar_modelo(raw, cfg["base_dir"])
        except Exception as exc:
            st.error(f"No se pudo cargar el modelo: {exc}")

    st.markdown("# 🏋️ Press Banca · Dashboard Multimodal")
    st.markdown(
        f'<p style="margin-top:-10px">IMU (MPU6050 + ESP32) + Pose (YOLOv8) · Random Forest · 4 clases · '
        f'<strong style="color:{t["accent"]}">Accuracy test {ACC_TEST:.2%}</strong> · '
        f'<strong style="color:{t["accent2"]}">F1 macro {F1_TEST:.3f}</strong></p>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="warn-card">⚠️ Faltan archivos: ajusta BASE_DIR en el sidebar '
        '(df_full_fusion.csv y modelo_press_banca_rf.joblib).</div>'
        if (df.empty or est is None) else "",
        unsafe_allow_html=True)
    st.markdown("---")

    render_kpis(cfg, est, feats)
    st.markdown("---")
    info = render_explorador(df, est, feats, idx, cfg)
    st.markdown("---")
    render_vision(info, cfg, idx)
    st.markdown("---")
    render_imu_fft(info, est, feats, cfg)
    st.markdown("---")
    render_diagnostico(info, cfg)


if __name__ == "__main__":
    main()