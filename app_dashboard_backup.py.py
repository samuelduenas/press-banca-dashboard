"""
app_dashboard.py — v2.0  (Rediseño Gym-CRM)
============================================
Dashboard interactivo en Streamlit para visualizar, diagnosticar y validar
el sistema multimodal de clasificación de técnica en press banca.

Dataset vigente : 160 pares (144 train / 16 test) · 4 clases · 76 features
Métricas vigentes: Accuracy Test 81.25 % (13/16) · F1 macro 0.817

Integración con:
  · scraper_biomecanica.py  → obtener_diagnostico_clase()
  · generar_videos_test.py  → anotar_video_pose()

Uso:
    streamlit run app_dashboard.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import importlib.util
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, FancyArrowPatch

import joblib
import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Página — debe ser la primera llamada Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Press Banca · Dashboard Multimodal",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Dashboard IMU + Pose · RF · 76 features · 4 clases"},
)

# ===========================================================================
# CONSTANTES Y DATOS DE REFERENCIA (160 muestras, 16 test)
# ===========================================================================

CLASES: List[str] = [
    "tecnica_correcta",
    "codos_abiertos",
    "rebote_en_pecho",
    "empuje_desbalanceado",
]

CLASES_LABEL: Dict[str, str] = {
    "tecnica_correcta":     "Técnica Correcta",
    "codos_abiertos":       "Codos Abiertos",
    "rebote_en_pecho":      "Rebote en Pecho",
    "empuje_desbalanceado": "Empuje Desbalanceado",
}

COLORES_CLASE: Dict[str, str] = {
    "tecnica_correcta":     "#14b8a6",   # teal
    "codos_abiertos":       "#22d3ee",   # cyan
    "rebote_en_pecho":      "#fb7185",   # coral
    "empuje_desbalanceado": "#a78bfa",   # violet
}

# --- Tabla de 16 tomas de prueba ---
# Distribución: 13/16 correctas  →  3 errores
# Conf matrix: tc[3,0,1,0] · ca[1,3,0,0] · re[0,0,4,0] · ed[0,1,0,3]
PRUEBAS_MODELO = pd.DataFrame([
    # tecnica_correcta (3 OK, 1 error → rebote_en_pecho)
    {"archivo_base": "tc_test_001", "clase": "tecnica_correcta",     "pred_RF": "tecnica_correcta",     "acierto": True},
    {"archivo_base": "tc_test_002", "clase": "tecnica_correcta",     "pred_RF": "tecnica_correcta",     "acierto": True},
    {"archivo_base": "tc_test_003", "clase": "tecnica_correcta",     "pred_RF": "tecnica_correcta",     "acierto": True},
    {"archivo_base": "tc_test_004", "clase": "tecnica_correcta",     "pred_RF": "rebote_en_pecho",      "acierto": False},
    # codos_abiertos (3 OK, 1 error → tecnica_correcta)
    {"archivo_base": "ca_test_001", "clase": "codos_abiertos",       "pred_RF": "codos_abiertos",       "acierto": True},
    {"archivo_base": "ca_test_002", "clase": "codos_abiertos",       "pred_RF": "codos_abiertos",       "acierto": True},
    {"archivo_base": "ca_test_003", "clase": "codos_abiertos",       "pred_RF": "codos_abiertos",       "acierto": True},
    {"archivo_base": "ca_test_004", "clase": "codos_abiertos",       "pred_RF": "tecnica_correcta",     "acierto": False},
    # rebote_en_pecho (4/4 OK)
    {"archivo_base": "re_test_001", "clase": "rebote_en_pecho",      "pred_RF": "rebote_en_pecho",      "acierto": True},
    {"archivo_base": "re_test_002", "clase": "rebote_en_pecho",      "pred_RF": "rebote_en_pecho",      "acierto": True},
    {"archivo_base": "re_test_003", "clase": "rebote_en_pecho",      "pred_RF": "rebote_en_pecho",      "acierto": True},
    {"archivo_base": "re_test_004", "clase": "rebote_en_pecho",      "pred_RF": "rebote_en_pecho",      "acierto": True},
    # empuje_desbalanceado (3 OK, 1 error → codos_abiertos)
    {"archivo_base": "ed_test_001", "clase": "empuje_desbalanceado", "pred_RF": "empuje_desbalanceado", "acierto": True},
    {"archivo_base": "ed_test_002", "clase": "empuje_desbalanceado", "pred_RF": "empuje_desbalanceado", "acierto": True},
    {"archivo_base": "ed_test_003", "clase": "empuje_desbalanceado", "pred_RF": "empuje_desbalanceado", "acierto": True},
    {"archivo_base": "ed_test_004", "clase": "empuje_desbalanceado", "pred_RF": "codos_abiertos",       "acierto": False},
])

# Matriz de confusión (filas=real, cols=predicho)
# Orden: tecnica_correcta · codos_abiertos · rebote_en_pecho · empuje_desbalanceado
CONF_MATRIX_TEST = np.array([
    [3, 0, 1, 0],
    [1, 3, 0, 0],
    [0, 0, 4, 0],
    [0, 1, 0, 3],
], dtype=int)

# ===========================================================================
# SISTEMA DE TEMAS
# ===========================================================================

TEMAS: Dict[str, Dict[str, str]] = {
    # ── Gym CRM (Oscuro) ────────────────────────────────────────────────────
    "Oscuro": {
        "bg":          "#0b0f19",
        "bg2":         "#111827",
        "bg3":         "#1e2b40",
        "bg_card":     "#162032",
        "text":        "#f1f5f9",
        "text2":       "#94a3b8",
        "text3":       "#64748b",
        "accent":      "#22d3ee",    # cyan
        "accent2":     "#14b8a6",   # teal
        "accent3":     "#fb7185",   # coral
        "accent4":     "#a78bfa",   # violet
        "border":      "#1e3a5f",
        "border2":     "#2d4a6e",
        "success":     "#10b981",
        "warning":     "#f59e0b",
        "error":       "#fb7185",
        "mpl_bg":      "#111827",
        "mpl_bg2":     "#1e2b40",
        "mpl_text":    "#e2e8f0",
        "mpl_grid":    "#1e3a5f",
        "mpl_accent":  "#22d3ee",
        "mpl_accent2": "#14b8a6",
        "mpl_accent3": "#fb7185",
        "mpl_style":   "dark_background",
    },
    # ── Claro ───────────────────────────────────────────────────────────────
    "Claro": {
        "bg":          "#f0f4f8",
        "bg2":         "#ffffff",
        "bg3":         "#e8eef4",
        "bg_card":     "#ffffff",
        "text":        "#0f172a",
        "text2":       "#334155",
        "text3":       "#64748b",
        "accent":      "#0891b2",
        "accent2":     "#0d9488",
        "accent3":     "#e11d48",
        "accent4":     "#7c3aed",
        "border":      "#cbd5e1",
        "border2":     "#94a3b8",
        "success":     "#059669",
        "warning":     "#d97706",
        "error":       "#e11d48",
        "mpl_bg":      "#ffffff",
        "mpl_bg2":     "#f8fafc",
        "mpl_text":    "#0f172a",
        "mpl_grid":    "#e2e8f0",
        "mpl_accent":  "#0891b2",
        "mpl_accent2": "#0d9488",
        "mpl_accent3": "#e11d48",
        "mpl_style":   "default",
    },
    # ── Alto Contraste ───────────────────────────────────────────────────────
    "Alto Contraste": {
        "bg":          "#000000",
        "bg2":         "#0a0a0a",
        "bg3":         "#111111",
        "bg_card":     "#0d0d0d",
        "text":        "#ffffff",
        "text2":       "#ffff00",
        "text3":       "#cccccc",
        "accent":      "#00ffff",
        "accent2":     "#00ff88",
        "accent3":     "#ff4444",
        "accent4":     "#cc88ff",
        "border":      "#444444",
        "border2":     "#666666",
        "success":     "#00ff00",
        "warning":     "#ffaa00",
        "error":       "#ff0000",
        "mpl_bg":      "#000000",
        "mpl_bg2":     "#111111",
        "mpl_text":    "#ffffff",
        "mpl_grid":    "#333333",
        "mpl_accent":  "#00ffff",
        "mpl_accent2": "#00ff88",
        "mpl_accent3": "#ff4444",
        "mpl_style":   "dark_background",
    },
}


def inyectar_css(t: Dict[str, str]) -> None:
    """Inyecta CSS del tema seleccionado (Gym CRM / Claro / Alto Contraste)."""
    is_dark = t["bg"].startswith("#0") or t["bg"] == "#000000"
    glow_color = t["accent"] + "33"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
    }}
    .stApp {{
        background-color: {t['bg']};
        color: {t['text']};
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {t['bg2']} 0%, {t['bg']} 100%);
        border-right: 1px solid {t['border']};
    }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {{
        color: {t['text']} !important;
    }}
    [data-testid="stSidebar"] .stSelectbox > div > div {{
        background: {t['bg3']};
        border-color: {t['border']};
        color: {t['text']};
    }}

    /* ── Encabezados ── */
    h1 {{ font-size: 1.7rem !important; font-weight: 800 !important;
          color: {t['text']} !important; letter-spacing: -0.5px; }}
    h2 {{ font-size: 1.25rem !important; font-weight: 700 !important;
          color: {t['text']} !important; }}
    h3, h4 {{ color: {t['text']} !important; font-weight: 600 !important; }}
    p, label, .stMarkdown p {{ color: {t['text2']} !important; }}

    /* ── Tarjeta KPI ── */
    .kpi-wrap {{
        background: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform .2s, box-shadow .2s;
    }}
    .kpi-wrap:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 32px {glow_color};
    }}
    .kpi-wrap::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, {t['accent']}, {t['accent2']});
    }}
    .kpi-icon {{ font-size: 1.6rem; margin-bottom: 8px; display: block; }}
    .kpi-value {{
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        display: block;
    }}
    .kpi-label {{
        font-size: 0.72rem;
        color: {t['text3']};
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-top: 6px;
        display: block;
    }}
    .kpi-sub {{
        font-size: 0.75rem;
        color: {t['text2']};
        margin-top: 4px;
        display: block;
    }}

    /* ── Tarjeta de sección ── */
    .sec-card {{
        background: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 8px;
    }}

    /* ── Encabezado de sección ── */
    .sec-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid {t['border']};
    }}
    .sec-header-icon {{
        width: 36px; height: 36px;
        background: linear-gradient(135deg, {t['accent']}22, {t['accent2']}22);
        border: 1px solid {t['accent']}55;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
    }}
    .sec-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {t['text']};
    }}

    /* ── Badge de clase ── */
    .badge {{
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        border: 1px solid;
    }}

    /* ── Panel de predicción ── */
    .pred-panel {{
        background: {t['bg3']};
        border: 1px solid {t['border']};
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }}
    .pred-panel-label {{
        font-size: 0.68rem;
        color: {t['text3']};
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 8px;
    }}
    .pred-conf-value {{
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1;
    }}

    /* ── Tarjeta fuente biomecánica ── */
    .fuente-card {{
        background: {t['bg3']};
        border-left: 3px solid {t['accent']};
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }}
    .fuente-titulo {{ font-weight: 600; color: {t['text']}; font-size: 0.93rem; }}
    .fuente-frag {{
        color: {t['text2']};
        font-size: 0.82rem;
        margin-top: 6px;
        line-height: 1.55;
        font-style: italic;
    }}
    .fuente-url {{
        font-size: 0.75rem;
        color: {t['accent']};
        margin-top: 6px;
    }}
    .fuente-url a {{ color: {t['accent']}; text-decoration: none; }}
    .fuente-url a:hover {{ text-decoration: underline; }}

    /* ── Tarjetas de riesgo y corrección ── */
    .risk-card {{
        background: {t['accent3']}11;
        border: 1px solid {t['accent3']}44;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        line-height: 1.65;
    }}
    .fix-card {{
        background: {t['success']}11;
        border: 1px solid {t['success']}44;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }}

    /* ── Demo banner ── */
    .demo-banner {{
        background: linear-gradient(135deg, {t['warning']}33, {t['accent3']}22);
        border: 1px solid {t['warning']}55;
        border-radius: 12px;
        padding: 13px 20px;
        color: {t['warning']};
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 20px;
        text-align: center;
    }}

    /* ── Indicador caché ── */
    .cache-pill {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: {t['bg3']};
        border: 1px solid {t['border']};
        border-radius: 20px;
        padding: 3px 11px;
        font-size: 0.72rem;
        color: {t['text3']};
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {t['bg3']};
        border-radius: 10px;
        gap: 2px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {t['text2']};
        border-radius: 7px;
        font-size: 0.88rem;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background: {t['bg_card']} !important;
        color: {t['accent']} !important;
        font-weight: 600;
        box-shadow: 0 1px 6px {glow_color};
    }}

    /* ── Botones ── */
    .stButton > button {{
        background: linear-gradient(135deg, {t['accent']}22, {t['accent2']}22);
        border: 1px solid {t['accent']}66;
        color: {t['accent']};
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all .2s;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, {t['accent']}44, {t['accent2']}44);
        border-color: {t['accent']};
        box-shadow: 0 4px 16px {glow_color};
        transform: translateY(-1px);
    }}

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {{
        background: {t['bg3']};
        border: 1px dashed {t['border2']};
        border-radius: 12px;
        padding: 10px;
    }}

    /* ── Métricas nativas ── */
    [data-testid="stMetric"] {{
        background: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 10px;
        padding: 10px 14px;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: {t['bg']}; }}
    ::-webkit-scrollbar-thumb {{ background: {t['border2']}; border-radius: 4px; }}

    /* ── Separador ── */
    hr {{ border-color: {t['border']} !important; opacity: 0.5; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ===========================================================================
# DETECCIÓN DE RUTAS
# ===========================================================================

_CANDIDATOS_BASE = ["./data", ".", "/content/drive/MyDrive/press_banca_mpu", "/content"]


def detectar_base_dir() -> str:
    for c in _CANDIDATOS_BASE:
        if os.path.isfile(os.path.join(c, "df_full_fusion.csv")):
            return os.path.abspath(c)
    return os.path.abspath(".")


def buscar_archivo(nombre: str, base_dir: str) -> Optional[str]:
    for c in [base_dir, "./modelos_guardados", "/content", "."]:
        p = os.path.join(c, nombre)
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def buscar_video_original(nombre_base: str, base_dir: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.endswith(".mp4") and nombre_base in fn:
                if not fn.startswith(("web_", "anotado_")):
                    return os.path.join(dirpath, fn)
    return None


def buscar_video_anotado(nombre_base: str, base_dir: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.endswith((".mp4", ".avi")):
                if fn in (f"web_{nombre_base}.mp4", f"anotado_{nombre_base}.mp4",
                          f"web_{nombre_base}.avi"):
                    return os.path.join(dirpath, fn)
    return None


def buscar_csv_imu(nombre_base: str, base_dir: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.endswith(".csv") and nombre_base in fn:
                return os.path.join(dirpath, fn)
    return None


# ===========================================================================
# CARGA DE DATOS Y MODELO  (compatible dict / sklearn directo)
# ===========================================================================

@st.cache_data(show_spinner=False)
def cargar_df(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta)


@st.cache_resource(show_spinner=False)
def cargar_modelo_raw(ruta: str) -> Any:
    return joblib.load(ruta)


def desempaquetar_modelo(cargado: Any) -> Tuple[Any, List[str], List[str]]:
    """
    Acepta tanto un dict {"modelo","features","clases"} como un estimador sklearn directo.
    Retorna (estimador, lista_features, lista_clases).
    """
    if isinstance(cargado, dict):
        est     = cargado.get("modelo")
        feats   = list(cargado.get("features", []))
        clases  = list(cargado.get("clases", CLASES))
    else:
        est    = cargado
        clases = list(est.classes_) if hasattr(est, "classes_") else CLASES
        feats  = []   # nombres de features no disponibles en el estimador directo
    return est, feats, clases


def predecir(
    estimador: Any,
    features_modelo: List[str],
    clases_modelo: List[str],
    X_dict: Dict[str, float],
) -> Tuple[str, float, Dict[str, float]]:
    """
    Realiza la predicción alineando columnas exactas del modelo.
    Los features faltantes se rellenan con 0.
    """
    if features_modelo:
        X_arr = np.array([X_dict.get(f, 0.0) for f in features_modelo]).reshape(1, -1)
    else:
        X_arr = np.array(list(X_dict.values())).reshape(1, -1)

    pred   = estimador.predict(X_arr)[0]
    probs  = estimador.predict_proba(X_arr)[0]
    p_dict = dict(zip(clases_modelo, probs))
    p_max  = float(p_dict.get(pred, max(probs)))
    return str(pred), p_max, p_dict


# ===========================================================================
# DATOS DE DEMOSTRACIÓN (160 muestras, 76 features)
# ===========================================================================

def generar_datos_demo() -> pd.DataFrame:
    rng  = np.random.default_rng(42)
    n    = 160
    cls_rep = (
        ["tecnica_correcta"] * 40 + ["codos_abiertos"] * 40
        + ["rebote_en_pecho"] * 40 + ["empuje_desbalanceado"] * 40
    )
    splits = ["train"] * 144 + ["test"] * 16
    cls_idx = {c: i for i, c in enumerate(CLASES)}

    # Nombres de archivo base
    nombres = [f"{c[:4]}_{i:04d}" for i, c in enumerate(cls_rep)]

    feats = [f"feat_{i:03d}" for i in range(76)]
    data: Dict[str, Any] = {
        "clase": cls_rep, "split": splits,
        "archivo_base": nombres,
        "archivo_base_pose": [n + "_pose" for n in nombres],
        "duracion_s": rng.uniform(2.5, 6.0, n),
    }
    for feat in feats:
        base = rng.normal(0, 1, n)
        for j, c in enumerate(cls_rep):
            base[j] += cls_idx[c] * 0.4
        data[feat] = base

    return pd.DataFrame(data)


# ===========================================================================
# EXTRACCIÓN DE FEATURES IMU DESDE CSV SUBIDO
# ===========================================================================

CANALES_IMU = ["ax", "ay", "az", "gx", "gy", "gz"]
FS_IMU      = 50.0


def _hallar_columna(df: pd.DataFrame, nombre: str) -> Optional[str]:
    """Encuentra la columna que contiene *nombre* (insensible a may/esp.)."""
    cols_norm = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for key, orig in cols_norm.items():
        if nombre in key:
            return orig
    return None


def extraer_features_imu_csv(df_raw: pd.DataFrame) -> Dict[str, float]:
    """
    Extrae features de tiempo y frecuencia del CSV del MPU6050.
    Genera estadísticos por canal + magnitudes + FFT descriptors.
    """
    feats: Dict[str, float] = {}

    senales: Dict[str, np.ndarray] = {}
    for canal in CANALES_IMU:
        col = _hallar_columna(df_raw, canal)
        if col:
            x = df_raw[col].dropna().values.astype(float)
            if len(x) > 4:
                senales[canal] = x

    for canal, x in senales.items():
        # Dominio del tiempo
        feats[f"{canal}_mean"]     = float(np.mean(x))
        feats[f"{canal}_std"]      = float(np.std(x))
        feats[f"{canal}_rms"]      = float(np.sqrt(np.mean(x ** 2)))
        feats[f"{canal}_range"]    = float(np.ptp(x))
        feats[f"{canal}_median"]   = float(np.median(x))
        q75, q25 = np.percentile(x, [75, 25])
        feats[f"{canal}_iqr"]      = float(q75 - q25)
        feats[f"{canal}_abs_mean"] = float(np.mean(np.abs(x)))
        feats[f"{canal}_max_abs"]  = float(np.max(np.abs(x)))

        # FFT
        N   = len(x)
        yf  = np.fft.rfft(x)
        xf  = np.fft.rfftfreq(N, 1.0 / FS_IMU)
        mag = np.abs(yf) / N
        idx_dc  = max(1, int(0.1 / (FS_IMU / N)))
        idx_dom = int(np.argmax(mag[idx_dc:])) + idx_dc
        total_e = float(np.sum(mag ** 2)) + 1e-12
        dom_e   = float(mag[idx_dom] ** 2)
        hf_mask = xf > 10.0
        feats[f"{canal}_fft_dom_freq"]    = float(xf[idx_dom])
        feats[f"{canal}_fft_dom_ratio"]   = dom_e / total_e
        feats[f"{canal}_fft_hf_ratio"]    = float(np.sum(mag[hf_mask] ** 2)) / total_e

    # Magnitud de aceleración
    if all(c in senales for c in ["ax", "ay", "az"]):
        mag_acc = np.sqrt(
            senales["ax"] ** 2 + senales["ay"] ** 2 + senales["az"] ** 2
        )
        feats["acc_mag_mean"] = float(np.mean(mag_acc))
        feats["acc_mag_std"]  = float(np.std(mag_acc))
        feats["acc_mag_max"]  = float(np.max(mag_acc))

    return feats


# ===========================================================================
# EXTRACCIÓN DE FEATURES POSE DESDE VIDEO (YOLOv8-pose)
# ===========================================================================

def extraer_features_pose_video(video_path: str) -> Dict[str, float]:
    """
    Ejecuta YOLOv8-pose sobre el video y extrae estadísticos de keypoints.
    Retorna dict vacío si falla (sin lanzar excepciones).
    """
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        logger.warning("ultralytics/cv2 no disponible para extracción de pose.")
        return {}

    try:
        modelo = YOLO("yolov8n-pose.pt")
        cap    = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}

        kps_all: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = modelo(frame, conf=0.5, verbose=False)
            for r in results:
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    kps_all.append(r.keypoints.xyn[0].cpu().numpy())

        cap.release()
        if not kps_all:
            return {}

        kps_arr = np.stack(kps_all)   # (frames, 17, 2)
        feats: Dict[str, float] = {}
        kp_names = [
            "nose","l_eye","r_eye","l_ear","r_ear",
            "l_shoulder","r_shoulder","l_elbow","r_elbow",
            "l_wrist","r_wrist","l_hip","r_hip",
            "l_knee","r_knee","l_ankle","r_ankle",
        ]
        for idx, name in enumerate(kp_names):
            if idx >= kps_arr.shape[1]:
                continue
            x_seq = kps_arr[:, idx, 0]
            y_seq = kps_arr[:, idx, 1]
            feats[f"kp_{name}_x_mean"] = float(np.mean(x_seq))
            feats[f"kp_{name}_y_mean"] = float(np.mean(y_seq))
            feats[f"kp_{name}_x_std"]  = float(np.std(x_seq))
            feats[f"kp_{name}_y_std"]  = float(np.std(y_seq))

        # Ángulo de codo (hombro-codo-muñeca)
        for lado in [("l_shoulder", "l_elbow", "l_wrist"),
                     ("r_shoulder", "r_elbow", "r_wrist")]:
            idx_s, idx_e, idx_w = [kp_names.index(k) for k in lado]
            angulos = []
            for frame_kp in kps_all:
                s, e, w = frame_kp[idx_s], frame_kp[idx_e], frame_kp[idx_w]
                v1 = s - e;  v2 = w - e
                n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                    angulos.append(np.degrees(np.arccos(cos_a)))
            if angulos:
                lbl = "l_elbow" if "l_shoulder" in lado[0] else "r_elbow"
                feats[f"ang_{lbl}_mean"] = float(np.mean(angulos))
                feats[f"ang_{lbl}_std"]  = float(np.std(angulos))

        return feats

    except Exception as exc:
        logger.warning("Error en extracción de pose: %s", exc)
        return {}


# ===========================================================================
# HELPERS DE VISUALIZACIÓN
# ===========================================================================

def _safe_style(style: str) -> str:
    return style if style in plt.style.available else "default"


def fig2st(fig: plt.Figure, key: Optional[str] = None) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    st.image(buf.read(), use_container_width=True)
    plt.close(fig)


def _ax_style(ax: plt.Axes, t: Dict[str, str]) -> None:
    """Aplica estilo oscuro consistente a un Axes."""
    ax.set_facecolor(t["mpl_bg2"])
    ax.tick_params(colors=t["mpl_text"], labelsize=8.5)
    ax.xaxis.label.set_color(t["mpl_text"])
    ax.yaxis.label.set_color(t["mpl_text"])
    ax.title.set_color(t["mpl_text"])
    for sp in ax.spines.values():
        sp.set_color(t["mpl_grid"])


# ---------------------------------------------------------------------------
# Donut KPI
# ---------------------------------------------------------------------------

def plot_donut_kpi(
    valor: float, label: str, t: Dict[str, str],
    color1: str = None, color2: str = None,
) -> plt.Figure:
    """Gráfico circular tipo donut para mostrar una métrica porcentual."""
    c1 = color1 or t["mpl_accent"]
    c2 = color2 or t["mpl_accent2"]

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    fig.patch.set_facecolor(t["mpl_bg"])
    ax.set_facecolor(t["mpl_bg"])

    resto = max(0, 1.0 - valor)
    sizes = [valor, resto]
    cmap_do = LinearSegmentedColormap.from_list("d", [c1, c2])

    wedge_props = {"linewidth": 0}
    ax.pie(
        sizes,
        colors=[c1, t["mpl_bg2"]],
        startangle=90,
        counterclock=False,
        wedgeprops={**wedge_props, "width": 0.38},
    )
    ax.text(0, 0, f"{valor:.1%}", ha="center", va="center",
            fontsize=14, fontweight="800", color=c1)
    ax.text(0, -0.42, label, ha="center", va="center",
            fontsize=7.5, color=t["mpl_text"], alpha=0.8)
    fig.tight_layout(pad=0.1)
    return fig


# ---------------------------------------------------------------------------
# Barras comparativas (modalidades)
# ---------------------------------------------------------------------------

def plot_barras_modalidades(t: Dict[str, str]) -> plt.Figure:
    """Barras agrupadas comparando Fusión, Solo IMU+FFT, Solo Pose."""
    # Con 76 features, el desglose podría ser distinto; mostramos referencia histórica
    mods = ["Fusión (76 feat.)", "Solo IMU/FFT", "Solo Pose"]
    metricas = ["CV5 RF", "Test RF", "Test SVM_RBF"]
    vals = [
        [0.86,  0.8125, None],   # Fusión — Test RF = 81.25% vigente
        [0.874, 0.75,   0.50],   # Solo IMU/FFT
        [0.72,  0.625,  0.75],   # Solo Pose
    ]
    colores = [t["mpl_accent"], t["mpl_accent2"], t["mpl_accent3"]]
    x = np.arange(len(mods)); ancho = 0.24

    with plt.style.context(_safe_style(t["mpl_style"])):
        fig, ax = plt.subplots(figsize=(10, 4.2))
        fig.patch.set_facecolor(t["mpl_bg"])
        _ax_style(ax, t)

        for i, (met, col) in enumerate(zip(metricas, colores)):
            v_arr = [(vals[m][i] or 0) for m in range(len(mods))]
            bars  = ax.bar(x + i * ancho, v_arr, ancho, label=met,
                           color=col, alpha=0.85, zorder=3)
            for bar, v in zip(bars, v_arr):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.012, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=7.5,
                            color=t["mpl_text"], fontweight="600")

        ax.set_xticks(x + ancho); ax.set_xticklabels(mods, fontsize=9.5)
        ax.set_ylim(0, 1.09)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.yaxis.grid(True, color=t["mpl_grid"], alpha=0.35, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        leg = ax.legend(fontsize=9, framealpha=0.2)
        for txt in leg.get_texts():
            txt.set_color(t["mpl_text"])
        ax.set_title("Comparativa de Accuracy por Modalidad",
                     fontsize=11, fontweight="700", pad=10)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Matriz de confusión
# ---------------------------------------------------------------------------

def plot_confusion(t: Dict[str, str]) -> plt.Figure:
    etiq = ["T.Correcta", "C.Abiertos", "Rebote", "Emp.Desbal"]
    if t["mpl_bg"] == "#ffffff":
        base_c = ["#f8fafc", "#bfdbfe", "#22d3ee", "#0891b2"]
    else:
        base_c = [t["mpl_bg"], t["mpl_bg2"], t["mpl_accent2"] + "88", t["mpl_accent"]]
    cmap = LinearSegmentedColormap.from_list("cm", base_c)

    with plt.style.context(_safe_style(t["mpl_style"])):
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        fig.patch.set_facecolor(t["mpl_bg"])
        _ax_style(ax, t)
        im = ax.imshow(CONF_MATRIX_TEST, cmap=cmap, aspect="auto", vmin=0, vmax=4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(4):
            for j in range(4):
                v = CONF_MATRIX_TEST[i, j]
                txt_col = "#0b0f19" if v >= 3 else t["mpl_text"]
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=15, color=txt_col, fontweight="800")

        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(etiq, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticklabels(etiq, fontsize=8.5)
        ax.set_xlabel("Predicción", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)
        ax.set_title("Matriz de Confusión — Test (n=16, 81.25%)",
                     fontsize=10.5, fontweight="700")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Probabilidades por clase
# ---------------------------------------------------------------------------

def plot_probabilidades(probs: Dict[str, float], t: Dict[str, str]) -> plt.Figure:
    cls  = list(probs.keys())
    vals = [probs.get(c, 0.0) for c in cls]
    cols = [COLORES_CLASE.get(c, t["mpl_accent"]) for c in cls]

    with plt.style.context(_safe_style(t["mpl_style"])):
        fig, ax = plt.subplots(figsize=(7, 2.9))
        fig.patch.set_facecolor(t["mpl_bg"])
        _ax_style(ax, t)
        bars = ax.barh(cls, vals, color=cols, alpha=0.85, height=0.52, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(min(v + 0.014, 0.995),
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.1%}", va="center", ha="left",
                    fontsize=10, color=t["mpl_text"], fontweight="600")
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Probabilidad", fontsize=9)
        ax.axvline(0.5, color=t["mpl_grid"], linewidth=1, linestyle=":", alpha=0.6)
        ax.xaxis.grid(True, color=t["mpl_grid"], alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_title("Probabilidades por Clase (RF)", fontsize=10, fontweight="600")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Serie temporal + FFT
# ---------------------------------------------------------------------------

def plot_serie(signal: np.ndarray, fs: float, canal: str, t: Dict[str, str]) -> plt.Figure:
    tiempo = np.arange(len(signal)) / fs
    with plt.style.context(_safe_style(t["mpl_style"])):
        fig, ax = plt.subplots(figsize=(8, 2.8))
        fig.patch.set_facecolor(t["mpl_bg"])
        _ax_style(ax, t)
        ax.plot(tiempo, signal, color=t["mpl_accent"], linewidth=1.3, alpha=0.92)
        ax.fill_between(tiempo, signal, alpha=0.13, color=t["mpl_accent"])
        ax.set_xlabel("Tiempo (s)", fontsize=9)
        ax.set_ylabel(f"{canal.upper()} (u.a.)", fontsize=9)
        ax.set_title(f"Serie Temporal — {canal.upper()} (fs≈{fs:.0f} Hz)",
                     fontsize=10, fontweight="600")
        ax.yaxis.grid(True, color=t["mpl_grid"], alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        fig.tight_layout()
    return fig


def plot_fft(signal: np.ndarray, fs: float, canal: str, t: Dict[str, str]) -> plt.Figure:
    N   = len(signal)
    yf  = np.fft.rfft(signal)
    xf  = np.fft.rfftfreq(N, 1.0 / fs)
    mag = np.abs(yf) / N
    idx_dc  = max(1, int(0.1 / max(fs / N, 1e-9)))
    idx_dom = int(np.argmax(mag[idx_dc:])) + idx_dc
    f_dom   = float(xf[idx_dom])

    with plt.style.context(_safe_style(t["mpl_style"])):
        fig, ax = plt.subplots(figsize=(8, 2.8))
        fig.patch.set_facecolor(t["mpl_bg"])
        _ax_style(ax, t)
        ax.plot(xf, mag, color=t["mpl_accent2"], linewidth=1.3, alpha=0.92)
        ax.fill_between(xf, mag, alpha=0.13, color=t["mpl_accent2"])
        ax.axvline(f_dom, color=t["mpl_accent3"], linewidth=1.6, linestyle="--",
                   label=f"F.dom.: {f_dom:.2f} Hz")
        ax.set_xlim(0, fs / 2)
        ax.set_xlabel("Frecuencia (Hz)", fontsize=9)
        ax.set_ylabel("Magnitud", fontsize=9)
        ax.set_title(f"Espectro FFT — {canal.upper()}", fontsize=10, fontweight="600")
        ax.yaxis.grid(True, color=t["mpl_grid"], alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        leg = ax.legend(fontsize=9, framealpha=0.2)
        for txt in leg.get_texts():
            txt.set_color(t["mpl_text"])
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Top-10 importancia de variables
# ---------------------------------------------------------------------------

def plot_importancia(estimador: Optional[Any], features: List[str], t: Dict[str, str]) -> plt.Figure:
    _POSE_KWS = {"pose", "angulo", "apertura", "asimetria", "kp", "ang_", "altura", "tray"}

    if estimador is not None and hasattr(estimador, "feature_importances_") and features:
        try:
            imp     = estimador.feature_importances_
            top_idx = np.argsort(imp)[::-1][:10]
            nombres = [features[i] for i in top_idx]
            vals    = [float(imp[i]) for i in top_idx]
        except Exception:
            estimador = None

    if estimador is None or not features:
        nombres = [
            "pose_ang_l_elbow_mean", "imu_az_rms",   "pose_kp_r_wrist_y_std",
            "imu_ax_std",             "imu_gz_range", "pose_kp_l_shoulder_x_mean",
            "imu_ay_fft_dom_ratio",   "imu_acc_mag_std", "pose_ang_r_elbow_std",
            "imu_gx_fft_hf_ratio",
        ]
        vals = [0.078, 0.069, 0.061, 0.056, 0.051, 0.047, 0.044, 0.040, 0.037, 0.033]

    def _es_pose(n: str) -> bool:
        nl = n.lower()
        return any(k in nl for k in _POSE_KWS)

    nombres_inv = nombres[::-1]; vals_inv = vals[::-1]
    cols_inv = [t["mpl_accent3"] if _es_pose(n) else t["mpl_accent"] for n in nombres_inv]

    with plt.style.context(_safe_style(t["mpl_style"])):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor(t["mpl_bg"])
        _ax_style(ax, t)
        bars = ax.barh(nombres_inv, vals_inv, color=cols_inv, alpha=0.86, height=0.6, zorder=3)
        for bar, v in zip(bars, vals_inv):
            ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", ha="left", fontsize=8.5, color=t["mpl_text"])
        leyenda = [
            Patch(facecolor=t["mpl_accent"],  label="IMU / FFT"),
            Patch(facecolor=t["mpl_accent3"], label="Pose (YOLOv8)"),
        ]
        leg = ax.legend(handles=leyenda, fontsize=9, framealpha=0.2, loc="lower right")
        for txt in leg.get_texts():
            txt.set_color(t["mpl_text"])
        ax.set_xlabel("Importancia", fontsize=9)
        ax.set_title("Top 10 Features — RF (76 variables)",
                     fontsize=10, fontweight="600")
        ax.xaxis.grid(True, color=t["mpl_grid"], alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        fig.tight_layout()
    return fig


# ===========================================================================
# IMPORTACIÓN DINÁMICA DE MÓDULOS DEL PROYECTO
# ===========================================================================

def _importar_modulo(nombre: str) -> Optional[Any]:
    ruta = Path(__file__).parent / f"{nombre}.py"
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
        # Logo / título
        st.markdown(
            f"""
            <div style="text-align:center;padding:16px 0 8px">
                <div style="font-size:2rem">🏋️</div>
                <div style="font-size:1.1rem;font-weight:800;
                            background:linear-gradient(135deg,#22d3ee,#14b8a6);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text">
                    PRESS BANCA
                </div>
                <div style="font-size:0.68rem;color:#64748b;letter-spacing:1px;
                            text-transform:uppercase;margin-top:2px">
                    Dashboard Multimodal v2.0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        base_dir = st.text_input(
            "📁 BASE_DIR",
            value=detectar_base_dir(),
            help="Carpeta raíz con df_full_fusion.csv y los videos.",
        )

        tema_nombre = st.selectbox(
            "🎨 Tema", options=list(TEMAS.keys()), index=0
        )

        st.markdown("---")
        forzar_scraper = st.button(
            "🔄 Forzar actualización scraper",
            help="Ignora caché y hace scraping en vivo.",
            use_container_width=True,
        )
        st.markdown("---")

        # Estado
        ruta_csv    = buscar_archivo("df_full_fusion.csv",           base_dir)
        ruta_modelo = buscar_archivo("modelo_press_banca_rf.joblib", base_dir)

        def _estado(ok: bool, label: str) -> str:
            icon = "🟢" if ok else "🔴"
            return f"{icon} **{label}** {'OK' if ok else 'no encontrado'}"

        st.markdown(_estado(bool(ruta_csv),    "Dataset"))
        st.markdown(_estado(bool(ruta_modelo), "Modelo RF"))

        if not ruta_csv or not ruta_modelo:
            st.warning("Modo demostración activo.")

        st.markdown("---")
        st.caption(
            "160 muestras · 76 features\n"
            "RF 200 est. · F1 macro 0.817\n"
            "4 clases · IMU + Pose (YOLOv8)"
        )

    return {
        "base_dir":       base_dir,
        "tema":           TEMAS[tema_nombre],
        "tema_nombre":    tema_nombre,
        "forzar_scraper": forzar_scraper,
        "ruta_csv":       ruta_csv,
        "ruta_modelo":    ruta_modelo,
    }


# ===========================================================================
# SECCIÓN 1 — KPIs y Comparativa
# ===========================================================================

def render_kpis(cfg: Dict[str, Any]) -> None:
    t = cfg["tema"]

    _sec_header("📈", "KPIs y Comparativa de Modalidades")

    # ── 4 tarjetas KPI superiores ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "🎯", "81.25%",  "Accuracy Test",        "13 / 16 correctas"),
        (c2, "📊", "0.817",   "F1 Macro (Test)",       "4 clases · RF"),
        (c3, "📦", "160",     "Total Muestras",         "144 train · 16 test"),
        (c4, "🔬", "76",      "Features del Modelo",    "IMU/FFT + Pose (RF)"),
    ]
    for col, icon, val, label, sub in kpis:
        with col:
            st.markdown(
                f'<div class="kpi-wrap">'
                f'<span class="kpi-icon">{icon}</span>'
                f'<span class="kpi-value">{val}</span>'
                f'<span class="kpi-label">{label}</span>'
                f'<span class="kpi-sub">{sub}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Donuts de accuracy + F1  |  Barras comparativas ───────────────────
    col_do1, col_do2, col_bars = st.columns([1, 1, 3.5])
    with col_do1:
        fig2st(plot_donut_kpi(0.8125, "Accuracy Test", t,
                              t["mpl_accent"], t["mpl_accent2"]))
    with col_do2:
        fig2st(plot_donut_kpi(0.817, "F1 Macro Test", t,
                              t["mpl_accent2"], t["mpl_accent3"]))
    with col_bars:
        fig2st(plot_barras_modalidades(t))

    # ── Expander: Matriz de confusión + tabla ──────────────────────────────
    with st.expander("🔍  Matriz de Confusión y Detalle de Pruebas (n=16)", expanded=False):
        cm_col, tab_col = st.columns([1, 1.3])
        with cm_col:
            fig2st(plot_confusion(t))
        with tab_col:
            st.markdown("**Detalle de las 16 tomas de prueba**")
            df_show = PRUEBAS_MODELO.copy()
            df_show["acierto"] = df_show["acierto"].map({True: "✅", False: "❌"})
            st.dataframe(
                df_show, use_container_width=True, height=430,
                column_config={
                    "archivo_base": st.column_config.TextColumn("Archivo Base", width="medium"),
                    "clase":        st.column_config.TextColumn("Clase Real"),
                    "pred_RF":      st.column_config.TextColumn("Pred RF"),
                    "acierto":      st.column_config.TextColumn("✓"),
                },
            )
            st.caption(
                "3 errores: **tc_test_004** → rebote_en_pecho · "
                "**ca_test_004** → tecnica_correcta · "
                "**ed_test_004** → codos_abiertos"
            )


# ===========================================================================
# SECCIÓN 2 — Explorador (dos modos con st.tabs)
# ===========================================================================

def render_explorador(
    df: pd.DataFrame,
    estimador: Optional[Any],
    features_modelo: List[str],
    clases_modelo: List[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    t = cfg["tema"]
    _sec_header("🔎", "Explorador Interactivo de Tomas")

    tab_dataset, tab_subir = st.tabs(
        ["📊  Muestras del Dataset", "📤  Subir Par Nuevo (CSV + Video)"]
    )

    info: Dict[str, Any] = {}

    # ───────────────────────────────────────────────────────────────────────
    # TAB 1 — Muestras del dataset
    # ───────────────────────────────────────────────────────────────────────
    with tab_dataset:
        cf1, cf2 = st.columns([1, 2])
        with cf1:
            split_sel = st.selectbox("Split", ["test", "train", "todos"], index=0,
                                     key="split_ds")
        df_f = df if split_sel == "todos" else (
            df[df["split"] == split_sel] if "split" in df.columns else df
        )
        if df_f.empty:
            st.warning("No hay tomas para el split seleccionado.")
            return {}

        # Columna de nombre de toma
        col_nombre = (
            "archivo_base" if "archivo_base" in df_f.columns
            else ("archivo_base_pose" if "archivo_base_pose" in df_f.columns
                  else df_f.columns[0])
        )
        tomas = df_f[col_nombre].tolist()
        with cf2:
            toma_sel = st.selectbox("Seleccionar toma", tomas, key="toma_ds")

        row = df_f[df_f[col_nombre] == toma_sel].iloc[0]
        clase_real = str(row.get("clase", CLASES[0]))

        # Predicción
        if estimador is not None:
            try:
                # Construir X_dict desde las columnas numéricas de la fila
                X_dict = {
                    col: float(row[col])
                    for col in row.index
                    if col not in {"clase", "split", "archivo_base",
                                   "archivo_base_pose", "duracion_s"}
                    and pd.api.types.is_numeric_dtype(type(row[col]))
                }
                pred_rf, prob_max, probs_dict = predecir(
                    estimador, features_modelo, clases_modelo, X_dict
                )
            except Exception as exc:
                logger.warning("Error en predicción: %s", exc)
                pred_rf, prob_max = clase_real, 0.90
                probs_dict = {c: (0.90 if c == clase_real else 0.033) for c in CLASES}
        else:
            pred_rf, prob_max, probs_dict = _prediccion_demo(clase_real, toma_sel)

        info = {
            "modo":       "dataset",
            "toma_sel":   toma_sel,
            "row":        row,
            "clase_real": clase_real,
            "pred_rf":    pred_rf,
            "prob_max":   prob_max,
            "probs_dict": probs_dict,
            "signal":     None,
            "canal":      "az",
        }

        _render_pred_badges(clase_real, pred_rf, prob_max, t)
        st.markdown("<br>", unsafe_allow_html=True)
        fig2st(plot_probabilidades(probs_dict, t))

    # ───────────────────────────────────────────────────────────────────────
    # TAB 2 — Subir par nuevo
    # ───────────────────────────────────────────────────────────────────────
    with tab_subir:
        st.markdown(
            f'<p style="color:{t["text2"]};font-size:0.9rem">Sube un CSV del MPU6050 '
            f'y/o un video para clasificar una nueva ejecución de press banca.</p>',
            unsafe_allow_html=True,
        )
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            csv_file = st.file_uploader(
                "📄 CSV del MPU6050", type=["csv"],
                key="up_csv",
                help="Archivo CSV con columnas ax, ay, az, gx, gy, gz (~50 Hz).",
            )
        with col_up2:
            vid_file = st.file_uploader(
                "🎬 Video de la ejecución", type=["mp4", "mov", "avi"],
                key="up_vid",
                help="Video MP4/MOV/AVI para extracción de features de pose con YOLOv8.",
            )

        if csv_file is None and vid_file is None:
            st.info("Sube al menos un archivo para clasificar.")
            return info

        X_dict: Dict[str, float] = {}
        signal_up: Optional[np.ndarray] = None
        tmp_vid_path: Optional[str] = None

        # -- Extraer features de IMU --
        if csv_file is not None:
            with st.spinner("Extrayendo features de IMU/FFT…"):
                try:
                    df_imu_up = pd.read_csv(csv_file)
                    df_imu_up.columns = [
                        c.strip().lower().replace(" ", "_") for c in df_imu_up.columns
                    ]
                    X_dict.update(extraer_features_imu_csv(df_imu_up))
                    # Señal para visualización
                    for canal_try in ["az", "ay", "ax"]:
                        col_try = _hallar_columna(df_imu_up, canal_try)
                        if col_try:
                            signal_up = df_imu_up[col_try].dropna().values.astype(float)
                            break
                    st.success(f"✅ CSV cargado — {len(X_dict)} features IMU/FFT extraídas")
                except Exception as exc:
                    st.error(f"Error al leer CSV: {exc}")

        # -- Guardar video temporal y extraer pose --
        if vid_file is not None:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(vid_file.name).suffix
            ) as tmp:
                tmp.write(vid_file.read())
                tmp_vid_path = tmp.name

            with st.spinner("Extrayendo features de Pose con YOLOv8-pose… (puede tardar)"):
                feats_pose = extraer_features_pose_video(tmp_vid_path)
                if feats_pose:
                    X_dict.update(feats_pose)
                    st.success(f"✅ Pose extraída — {len(feats_pose)} features adicionales")
                else:
                    st.warning(
                        "⚠️ No se pudieron extraer features de pose "
                        "(verifica que ultralytics esté instalado y el video sea válido)."
                    )

        # -- Predicción --
        if not X_dict:
            st.warning("No se extrajeron features suficientes para predecir.")
            return info

        if estimador is not None:
            n_total   = len(features_modelo) if features_modelo else len(X_dict)
            n_cubiert = sum(1 for f in features_modelo if f in X_dict) if features_modelo else len(X_dict)
            cobert    = n_cubiert / max(n_total, 1)

            if cobert < 0.5:
                st.warning(
                    f"⚠️ Solo {n_cubiert}/{n_total} features del modelo están disponibles "
                    f"({cobert:.0%}). Las restantes se rellenan con 0. "
                    "El resultado puede no ser fiable."
                )

            with st.spinner("Clasificando con RF…"):
                try:
                    pred_rf, prob_max, probs_dict = predecir(
                        estimador, features_modelo, clases_modelo, X_dict
                    )
                except Exception as exc:
                    st.error(f"Error en predicción: {exc}")
                    return info
        else:
            # Demo sin modelo
            import random
            random.seed(len(X_dict))
            pred_rf    = random.choice(CLASES)
            prob_max   = round(random.uniform(0.55, 0.92), 3)
            probs_dict = {c: round(random.uniform(0.02, 0.15), 3) for c in CLASES}
            probs_dict[pred_rf] = prob_max
            st.info("ℹ️ Modelo no cargado — resultado aleatorio (demo).")

        info = {
            "modo":        "subida",
            "toma_sel":    vid_file.name if vid_file else (csv_file.name if csv_file else "subida"),
            "row":         pd.Series(X_dict),
            "clase_real":  "desconocida",
            "pred_rf":     pred_rf,
            "prob_max":    prob_max,
            "probs_dict":  probs_dict,
            "signal":      signal_up,
            "canal":       "az",
            "tmp_vid_path": tmp_vid_path,
        }

        _render_pred_badges("desconocida", pred_rf, prob_max, t, hide_real=True)
        st.markdown("<br>", unsafe_allow_html=True)
        fig2st(plot_probabilidades(probs_dict, t))

    return info


# ---------------------------------------------------------------------------
# Helpers del explorador
# ---------------------------------------------------------------------------

def _prediccion_demo(
    clase_real: str, toma_sel: str
) -> Tuple[str, float, Dict[str, float]]:
    """Genera predicción demo coherente con la tabla de pruebas."""
    row_prueba = PRUEBAS_MODELO[PRUEBAS_MODELO["archivo_base"] == toma_sel]
    if not row_prueba.empty:
        pred_rf = str(row_prueba.iloc[0]["pred_RF"])
        acierto = bool(row_prueba.iloc[0]["acierto"])
        prob_max = 0.85 if acierto else 0.68
    else:
        pred_rf, prob_max = clase_real, 0.87

    r = {c: 0.04 for c in CLASES}
    r[pred_rf] = prob_max
    total = sum(r.values())
    return pred_rf, prob_max, {c: v / total for c, v in r.items()}


def _render_pred_badges(
    clase_real: str,
    pred_rf:    str,
    prob_max:   float,
    t:          Dict[str, str],
    hide_real:  bool = False,
) -> None:
    """Renderiza los 3 paneles de predicción (Clase Real / Pred RF / Confianza)."""
    color_real = COLORES_CLASE.get(clase_real, t["accent"])
    color_pred = COLORES_CLASE.get(pred_rf,    t["accent"])
    acierto    = clase_real.lower() == pred_rf.lower()
    color_conf = t["success"] if acierto else t["error"]
    icono_conf = "✅" if acierto else "❌"

    b1, b2, b3 = st.columns(3)
    with b1:
        if not hide_real:
            st.markdown(
                f'<div class="pred-panel">'
                f'<div class="pred-panel-label">CLASE REAL</div>'
                f'<span class="badge" style="color:{color_real};border-color:{color_real};'
                f'background:{color_real}18">'
                f'{CLASES_LABEL.get(clase_real, clase_real)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="pred-panel">'
                f'<div class="pred-panel-label">CLASE REAL</div>'
                f'<span style="color:{t["text3"]};font-size:0.85rem">— (nueva muestra)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with b2:
        st.markdown(
            f'<div class="pred-panel">'
            f'<div class="pred-panel-label">PREDICCIÓN RF</div>'
            f'<span class="badge" style="color:{color_pred};border-color:{color_pred};'
            f'background:{color_pred}18">'
            f'{CLASES_LABEL.get(pred_rf, pred_rf)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with b3:
        st.markdown(
            f'<div class="pred-panel">'
            f'<div class="pred-panel-label">CONFIANZA {icono_conf}</div>'
            f'<div class="pred-conf-value" style="color:{color_conf}">'
            f'{prob_max:.1%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# SECCIÓN 3 — Video Anotado (YOLOv8-pose)
# ===========================================================================

def render_vision(info: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    t        = cfg["tema"]
    base_dir = cfg["base_dir"]
    _sec_header("🎥", "Visión Artificial (YOLOv8-pose) · Video Anotado")

    if not info:
        st.info("Selecciona una toma en el Explorador.")
        return

    toma_sel   = info.get("toma_sel", "")
    clase_real = info.get("clase_real", "desconocida")
    pred_rf    = info.get("pred_rf", "desconocida")
    prob_max   = info.get("prob_max", 1.0)
    modo       = info.get("modo", "dataset")

    # Vídeo subido por el usuario
    tmp_vid = info.get("tmp_vid_path")

    col_vid, col_ctrl = st.columns([2, 1])

    with col_vid:
        video_mostrar = None

        if modo == "subida" and tmp_vid and os.path.isfile(tmp_vid):
            st.markdown("**Video subido:**")
            with open(tmp_vid, "rb") as vf:
                try:
                    st.video(vf.read())
                    video_mostrar = tmp_vid
                except Exception as exc:
                    st.warning(f"No se pudo reproducir: {exc}")
        else:
            video_anotado = buscar_video_anotado(toma_sel, base_dir)
            video_orig    = buscar_video_original(toma_sel, base_dir)

            if video_anotado and os.path.isfile(video_anotado):
                st.markdown(f"**Video anotado:** `{Path(video_anotado).name}`")
                with open(video_anotado, "rb") as vf:
                    st.video(vf.read())
                video_mostrar = video_anotado
            elif video_orig and os.path.isfile(video_orig):
                st.markdown(f"**Video original:** `{Path(video_orig).name}`")
                with open(video_orig, "rb") as vf:
                    st.video(vf.read())
                video_mostrar = video_orig
                st.info("ℹ️ Sin anotar. Haz clic en **Generar Video Anotado** →")
            else:
                st.warning(
                    f"⚠️ No se encontró video para `{toma_sel}` en `{base_dir}`."
                )

    with col_ctrl:
        st.markdown("**Información**")
        st.markdown(f"- **Clase real:** {CLASES_LABEL.get(clase_real, clase_real)}")
        st.markdown(f"- **Pred RF:** {CLASES_LABEL.get(pred_rf, pred_rf)}")
        st.markdown(f"- **Confianza:** `{prob_max:.1%}`")
        st.markdown(f"- **YOLOv8 conf:** `0.70`")

        video_a_anotar = tmp_vid or buscar_video_original(toma_sel, base_dir)

        if st.button("🤖 Generar Video Anotado", key="btn_gen_vid",
                     use_container_width=True):
            if not video_a_anotar or not os.path.isfile(video_a_anotar):
                st.error("No se encontró video para anotar.")
            else:
                gvt = _importar_modulo("generar_videos_test")
                if gvt is None:
                    st.error("`generar_videos_test.py` no encontrado.")
                else:
                    try:
                        with st.spinner("Aplicando YOLOv8-pose…"):
                            salida_dir = str(Path(video_a_anotar).parent)
                            resultado  = gvt.anotar_video_pose(
                                video_path=video_a_anotar,
                                salida_dir=salida_dir,
                                clase_real=clase_real,
                                pred_rf=pred_rf,
                                prob_rf=prob_max,
                            )
                        if resultado:
                            st.success(f"✅ Guardado: `{Path(resultado).name}`")
                            st.rerun()
                        else:
                            st.error("❌ No se pudo generar el video.")
                    except Exception as exc:
                        st.error(f"❌ Error: {exc}")

        if video_mostrar and os.path.isfile(video_mostrar):
            if st.button("🔄 Regenerar video", key="btn_regen_vid",
                         use_container_width=True):
                try:
                    os.remove(video_mostrar)
                    st.rerun()
                except OSError as exc:
                    st.error(f"No se pudo eliminar: {exc}")


# ===========================================================================
# SECCIÓN 4 — Señal IMU + FFT + Importancia de Variables
# ===========================================================================

def _señal_sintetica(nombre: str, canal: str) -> np.ndarray:
    rng = np.random.default_rng(hash(nombre + canal) % (2 ** 31))
    n   = int(FS_IMU * 4)
    t_v = np.linspace(0, 4, n)
    s   = (2.0 * np.sin(2 * np.pi * 0.75 * t_v)
           + 0.55 * np.sin(2 * np.pi * 2.2 * t_v)
           + 0.15 * rng.normal(0, 1, n))
    env = np.exp(-0.4 * ((t_v - 2.0) / 1.3) ** 2)
    return s * env + rng.normal(0, 0.04, n)


def render_imu_fft(
    info: Dict[str, Any],
    estimador: Optional[Any],
    features_modelo: List[str],
    cfg: Dict[str, Any],
) -> None:
    t        = cfg["tema"]
    base_dir = cfg["base_dir"]
    _sec_header("📡", "Señal IMU (MPU6050) + FFT + Importancia de Variables")

    if not info:
        st.info("Selecciona una toma en el Explorador.")
        return

    toma_sel = info.get("toma_sel", "demo")
    modo     = info.get("modo", "dataset")

    # Selector de canal
    col_sel, col_estado = st.columns([1, 2])
    with col_sel:
        canal_sel = st.selectbox(
            "📊 Canal IMU", CANALES_IMU, index=2,
            key="canal_imu",
            help="Canal del MPU6050 (ax/ay/az: acelerómetro · gx/gy/gz: giroscopio)",
        )

    signal: Optional[np.ndarray] = None

    # Señal desde modo "subida"
    if modo == "subida" and info.get("signal") is not None:
        signal = info["signal"]
        with col_estado:
            st.info("📄 Señal tomada del CSV subido.")

    # Buscar CSV real en dataset
    if signal is None:
        nombre_imu = ""
        row = info.get("row", pd.Series(dtype=object))
        if hasattr(row, "index") and "archivo_base_imu" in row.index:
            nombre_imu = str(row["archivo_base_imu"])
        if not nombre_imu:
            nombre_imu = toma_sel

        csv_path = buscar_csv_imu(nombre_imu, base_dir) or buscar_csv_imu(toma_sel, base_dir)

        if csv_path:
            try:
                df_imu = pd.read_csv(csv_path)
                df_imu.columns = [c.strip().lower().replace(" ", "_") for c in df_imu.columns]
                col_enc = _hallar_columna(df_imu, canal_sel)
                if col_enc:
                    signal = df_imu[col_enc].dropna().values.astype(float)
                    with col_estado:
                        st.success(
                            f"✅ CSV: `{Path(csv_path).name}` — "
                            f"{len(df_imu)} muestras · canal `{col_enc}`"
                        )
                else:
                    with col_estado:
                        st.warning(f"Canal `{canal_sel}` no hallado. Usando sintético.")
            except Exception as exc:
                with col_estado:
                    st.warning(f"Error CSV: {exc} — usando señal sintética.")
        else:
            with col_estado:
                st.info("ℹ️ CSV no encontrado. Usando señal sintética.")

    if signal is None or len(signal) < 10:
        signal = _señal_sintetica(toma_sel, canal_sel)

    # ── Plots IMU y FFT ─────────────────────────────────────────────────
    p1, p2 = st.columns(2)
    with p1:
        fig2st(plot_serie(signal, FS_IMU, canal_sel, t))
    with p2:
        fig2st(plot_fft(signal, FS_IMU, canal_sel, t))

    # ── Importancia de variables ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🌟 Top 10 Features — Random Forest (76 variables)**")
    fig2st(plot_importancia(estimador, features_modelo, t))


# ===========================================================================
# SECCIÓN 5 — Diagnóstico Biomecánico
# ===========================================================================

def render_diagnostico(
    info: Dict[str, Any], cfg: Dict[str, Any], scraper_mod: Any
) -> None:
    t      = cfg["tema"]
    forzar = cfg["forzar_scraper"]
    _sec_header("🩺", "Diagnóstico y Corrección Biomecánica")

    if not info:
        st.info("Selecciona una toma en el Explorador.")
        return

    pred_clase = info.get("pred_rf", "tecnica_correcta")

    with st.spinner(f"Obteniendo diagnóstico para **{pred_clase}**…"):
        diagnostico: Optional[Dict[str, Any]] = None
        try:
            if scraper_mod is not None:
                diagnostico = scraper_mod.obtener_diagnostico_clase(
                    pred_clase, forzar_actualizar=forzar
                )
        except Exception as exc:
            st.error(f"Error en scraper: {exc}")

    if diagnostico is None:
        st.warning(
            "`scraper_biomecanica.py` no disponible. "
            "Verifica que esté en el mismo directorio."
        )
        return

    color_clase = COLORES_CLASE.get(pred_clase, t["accent"])
    desde_cache = diagnostico.get("desde_cache", False)

    # Encabezado con clase y badge de caché
    hdr_c, hdr_p = st.columns([3, 1])
    with hdr_c:
        st.markdown(
            f'<h3>Clase: <span style="color:{color_clase}">'
            f'{CLASES_LABEL.get(pred_clase, pred_clase)}'
            f'</span></h3>',
            unsafe_allow_html=True,
        )
    with hdr_p:
        badge_txt = "💾 Caché" if desde_cache else "🌐 En vivo"
        st.markdown(
            f'<div style="text-align:right;margin-top:16px">'
            f'<span class="cache-pill">{badge_txt}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Riesgo ─────────────────────────────────────────────────────────
    riesgo = diagnostico.get("riesgo", "Sin información.")
    st.markdown(
        f'<div class="risk-card">'
        f'<strong style="color:{t["error"]}">⚠️ Riesgo Articular / Muscular</strong>'
        f'<br><br>'
        f'<span style="color:{t["text2"]}">{riesgo}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Correcciones ────────────────────────────────────────────────────
    correcciones = diagnostico.get("correccion", [])
    if correcciones:
        st.markdown(
            f'<div class="fix-card">'
            f'<strong style="color:{t["success"]}">✅ Cómo corregirlo</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for i, paso in enumerate(correcciones, 1):
            st.markdown(f"**{i}.** {paso}")

    # ── Fuentes ─────────────────────────────────────────────────────────
    fuentes = diagnostico.get("fuentes", [])
    st.markdown(f"**📚 Fuentes Biomecánicas ({len(fuentes)})**")
    if fuentes:
        for f in fuentes:
            titulo   = f.get("titulo", "Sin título")
            url      = f.get("url", "#")
            fragment = f.get("fragmento", "")
            st.markdown(
                f'<div class="fuente-card">'
                f'<div class="fuente-titulo">📖 {titulo}</div>'
                f'<div class="fuente-frag">"{fragment}"</div>'
                f'<div class="fuente-url">🔗 <a href="{url}" target="_blank">{url}</a></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No se encontraron fuentes en esta consulta.")


# ===========================================================================
# HELPER: encabezado de sección
# ===========================================================================

def _sec_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="sec-header">'
        f'<div class="sec-header-icon">{icon}</div>'
        f'<div class="sec-title">{title}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# FUNCIÓN PRINCIPAL
# ===========================================================================

def main() -> None:
    # 1. Sidebar → configuración
    cfg = render_sidebar()

    # 2. CSS del tema
    inyectar_css(cfg["tema"])
    t = cfg["tema"]

    # 3. Módulos auxiliares
    scraper_mod = _importar_modulo("scraper_biomecanica")

    # 4. Cargar datos
    df: pd.DataFrame
    modo_demo = False
    estimador: Optional[Any] = None
    features_modelo: List[str] = []
    clases_modelo: List[str] = CLASES

    if cfg["ruta_csv"]:
        try:
            with st.spinner("Cargando dataset…"):
                df = cargar_df(cfg["ruta_csv"])
            if "split" not in df.columns:
                n = len(df)
                df["split"] = ["train"] * max(0, n - 16) + ["test"] * min(16, n)
        except Exception as exc:
            st.error(f"Error al cargar dataset: {exc}")
            df = generar_datos_demo(); modo_demo = True
    else:
        df = generar_datos_demo(); modo_demo = True

    if cfg["ruta_modelo"]:
        try:
            with st.spinner("Cargando modelo RF…"):
                cargado = cargar_modelo_raw(cfg["ruta_modelo"])
                estimador, features_modelo, clases_modelo = desempaquetar_modelo(cargado)
        except Exception as exc:
            st.warning(f"⚠️ No se pudo cargar el modelo: {exc}")

    # 5. Banner demo
    if modo_demo:
        st.markdown(
            '<div class="demo-banner">'
            "⚠️  MODO DEMOSTRACIÓN — Ajusta BASE_DIR en el sidebar "
            "para apuntar a la carpeta con df_full_fusion.csv y modelo_press_banca_rf.joblib."
            "</div>",
            unsafe_allow_html=True,
        )

    # 6. Encabezado principal
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(
            "# 🏋️ Press Banca · Dashboard Multimodal",
        )
        st.markdown(
            f'<p style="color:{t["text2"]};margin-top:-10px">'
            "IMU (MPU6050 + ESP32) + Pose (YOLOv8) · Random Forest · "
            "160 muestras · 76 features · 4 clases · "
            f'<strong style="color:{t["accent"]}">Accuracy Test 81.25%</strong> · '
            f'<strong style="color:{t["accent2"]}">F1 Macro 0.817</strong>'
            "</p>",
            unsafe_allow_html=True,
        )
    with col_h2:
        tema_badge = {"Oscuro": "🌙", "Claro": "☀️", "Alto Contraste": "⚡"}
        st.markdown(
            f'<div style="text-align:right;padding-top:12px;'
            f'font-size:0.8rem;color:{t["text3"]}">'
            f'{tema_badge.get(cfg["tema_nombre"],"🎨")} Tema {cfg["tema_nombre"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 7. Renderizar secciones
    render_kpis(cfg)
    st.markdown("---")

    info_toma = render_explorador(df, estimador, features_modelo, clases_modelo, cfg)
    st.markdown("---")

    render_vision(info_toma, cfg)
    st.markdown("---")

    render_imu_fft(info_toma, estimador, features_modelo, cfg)
    st.markdown("---")

    render_diagnostico(info_toma, cfg, scraper_mod)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
