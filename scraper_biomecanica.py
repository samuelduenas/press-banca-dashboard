"""
scraper_biomecanica.py
======================
Módulo de web scraping biomecánico para el dashboard de clasificación de
técnica en press banca.

Función principal
-----------------
    obtener_diagnostico_clase(pred_clase, forzar_actualizar=False) -> dict

El dict retornado siempre tiene las claves:
    "clase"        : str
    "riesgo"       : str   – riesgo articular / muscular del fallo
    "correccion"   : list  – pasos concretos para corregir
    "fuentes"      : list  – [{"titulo": str, "url": str, "fragmento": str}]
    "desde_cache"  : bool  – True si el resultado vino del JSON en disco

Estrategia de datos
-------------------
1. Consulta caché JSON local (TTL 24 h).
2. Si la caché expiró o no existe → intenta scraping real (Stronger By Science
   y ExRx.net) filtrando párrafos por palabras clave de cada clase.
3. Si la red falla o el HTML no contiene párrafos relevantes → devuelve el
   diccionario biomecánico local sin lanzar excepciones.
"""

import json
import os
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

CACHE_FILE = "base_conocimiento_press.json"
CACHE_TTL_HORAS = 24          # horas de vida de cada entrada en caché
TIMEOUT_HTTP = 8              # segundos de timeout por petición

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

# ---------------------------------------------------------------------------
# Palabras clave de búsqueda por clase
# ---------------------------------------------------------------------------

KEYWORDS_POR_CLASE: Dict[str, List[str]] = {
    "codos_abiertos": [
        "elbow flare", "flared elbows", "bench press elbow",
        "shoulder injury bench press", "pectoral stress elbow angle",
        "elbows tucked", "elbow position bench",
    ],
    "rebote_en_pecho": [
        "bounce chest bench press", "touch and go", "paused bench press",
        "chest bounce injury", "bar path bounce", "pectoral tear bounce",
        "dead stop bench",
    ],
    "empuje_desbalanceado": [
        "uneven bench press", "asymmetric push", "bar tilt bench press",
        "unilateral imbalance bench", "one side stronger bench",
        "rotator cuff imbalance", "shoulder asymmetry press",
    ],
    "tecnica_correcta": [
        "proper bench press technique", "correct bench form",
        "retract scapula bench", "arch back bench press",
        "leg drive bench press", "bar path bench press optimal",
        "wrist position bench press",
    ],
}

# URLs de scraping por clase
URLS_POR_CLASE: Dict[str, List[str]] = {
    "codos_abiertos": [
        "https://www.strongerbyscience.com/how-to-bench/",
        "https://exrx.net/WeightExercises/PectoralSternal/BBBenchPress",
    ],
    "rebote_en_pecho": [
        "https://www.strongerbyscience.com/how-to-bench/",
        "https://exrx.net/WeightExercises/PectoralSternal/BBBenchPress",
    ],
    "empuje_desbalanceado": [
        "https://www.strongerbyscience.com/how-to-bench/",
        "https://exrx.net/WeightExercises/PectoralSternal/BBBenchPress",
    ],
    "tecnica_correcta": [
        "https://www.strongerbyscience.com/how-to-bench/",
        "https://exrx.net/WeightExercises/PectoralSternal/BBBenchPress",
    ],
}

# ---------------------------------------------------------------------------
# Diccionario biomecánico local – fallback garantizado
# ---------------------------------------------------------------------------

FALLBACK_LOCAL: Dict[str, Dict[str, Any]] = {
    "codos_abiertos": {
        "clase": "codos_abiertos",
        "riesgo": (
            "La apertura excesiva de codos (>75° respecto al torso) incrementa "
            "el momento de fuerza sobre el hombro y reduce la estabilidad "
            "glenohumeral. Esto sobrecarga los tendones del manguito rotador "
            "(especialmente el supraespinoso) y el labrum anterior, aumentando "
            "el riesgo de impingement subacromial, tendinitis crónica y, en "
            "casos extremos, desgarro del tendón del pectoral mayor."
        ),
        "correccion": [
            "Mantén los codos entre 45° y 60° respecto al torso durante todo "
            "el recorrido; evita sobrepasar los 75°.",
            "Activa el dorsal ancho ('aprieta naranjas en las axilas') para "
            "estabilizar la articulación glenohumeral en el descenso.",
            "Agarra la barra con la muñeca en posición neutra (alineada con "
            "el antebrazo) y coloca las manos ligeramente más anchas que los hombros.",
            "Practica la técnica con carga ligera frente a un espejo lateral "
            "o grábate para verificar el ángulo de codo en el punto más bajo.",
            "Realiza trabajo de movilidad de pectoral y rotadores externos "
            "(face pulls, band pull-aparts) para equilibrar la musculatura.",
        ],
        "fuentes": [
            {
                "titulo": "How to Bench Press – Stronger By Science",
                "url": "https://www.strongerbyscience.com/how-to-bench/",
                "fragmento": (
                    "Elbow flare increases shoulder stress significantly. "
                    "Keeping elbows at 45–60° reduces rotator cuff load and "
                    "allows for a stronger, safer press."
                ),
            },
            {
                "titulo": "Barbell Bench Press – ExRx.net",
                "url": "https://exrx.net/WeightExercises/PectoralSternal/BBBenchPress",
                "fragmento": (
                    "Avoid excessive elbow flare to minimize shoulder joint "
                    "stress. Tucking the elbows slightly protects the "
                    "glenohumeral joint and maintains pectoral activation."
                ),
            },
        ],
        "desde_cache": False,
    },
    "rebote_en_pecho": {
        "clase": "rebote_en_pecho",
        "riesgo": (
            "El rebote de la barra en el pecho es un patrón de movimiento que "
            "elimina el estímulo muscular en el punto de mayor ventaja mecánica "
            "del pectoral y triceps. Además, el impacto repetido sobre el "
            "esternón y las costillas puede provocar contusiones, fracturas por "
            "estrés en las uniones condrocostales y distensión del ligamento "
            "esternoclavicular. Biomecánicamente, la carga inercial en la "
            "deceleración activa los tejidos pasivos (ligamentos, cápsula "
            "articular) en lugar de la musculatura activa."
        ),
        "correccion": [
            "Adopta la variante 'paused bench': detén la barra 1–2 segundos "
            "sobre el pecho antes de empujar, eliminando la ayuda elástica.",
            "Controla la fase excéntrica (descenso) a un tempo de 2–3 segundos "
            "para aumentar el tiempo bajo tensión y mejorar el control neuromuscular.",
            "Toca el pecho de forma suave y controlada; la barra debe detenerse "
            "sin impacto, justo por debajo de la línea de los pezones.",
            "Reduce la carga entre un 10–15% mientras reeducas el patrón motor "
            "con el tempo correcto.",
            "Programa series de 'touch-and-go controlado' antes de retomar "
            "pesos máximos.",
        ],
        "fuentes": [
            {
                "titulo": "How to Bench Press – Stronger By Science",
                "url": "https://www.strongerbyscience.com/how-to-bench/",
                "fragmento": (
                    "Bouncing the bar off the chest removes muscle tension at "
                    "the bottom, reduces training stimulus, and shifts load to "
                    "passive structures. Paused bench press is recommended for "
                    "technique correction and hypertrophy."
                ),
            },
            {
                "titulo": "Barbell Bench Press – ExRx.net",
                "url": "https://exrx.net/WeightExercises/PectoralSternal/BBBenchPress",
                "fragmento": (
                    "Do not bounce the bar off the chest. Lower the bar under "
                    "control to the chest before pressing upward to maintain "
                    "muscular tension throughout the movement."
                ),
            },
        ],
        "desde_cache": False,
    },
    "empuje_desbalanceado": {
        "clase": "empuje_desbalanceado",
        "riesgo": (
            "Un empuje asimétrico indica una diferencia de fuerza, activación "
            "o movilidad entre el hemicuerpo derecho e izquierdo. Esto genera "
            "rotación involuntaria de la barra, compensaciones en la columna "
            "torácica y lumbar, y sobrecarga unilateral del hombro y el codo. "
            "A largo plazo puede producir lesiones de labrum, epicondilitis "
            "y desequilibrios posturales."
        ),
        "correccion": [
            "Realiza press con mancuernas unilateral (1 brazo) para identificar "
            "y trabajar el lado más débil de forma independiente.",
            "Verifica la posición y simetría de los pies: ambos pies deben "
            "ejercer la misma presión sobre el suelo (leg drive simétrico).",
            "Revisa el agarre: asegúrate de que la barra esté centrada usando "
            "los anillos de la barra como referencia visual.",
            "Practica el press con barra en rack de seguridad a media altura "
            "para detectar si la barra se inclina durante la fase concéntrica.",
            "Añade trabajo de core antirotación (pallof press) para mejorar la "
            "estabilidad del tronco durante el empuje bilateral.",
            "Consulta con un fisioterapeuta si la asimetría persiste: puede "
            "indicar inhibición neuromuscular o restricción articular.",
        ],
        "fuentes": [
            {
                "titulo": "How to Bench Press – Stronger By Science",
                "url": "https://www.strongerbyscience.com/how-to-bench/",
                "fragmento": (
                    "Uneven pressing often reflects a strength imbalance or "
                    "asymmetric setup. Single-arm dumbbell work and careful "
                    "attention to grip symmetry can resolve most cases."
                ),
            },
            {
                "titulo": "Barbell Bench Press – ExRx.net",
                "url": "https://exrx.net/WeightExercises/PectoralSternal/BBBenchPress",
                "fragmento": (
                    "Ensure the bar is evenly gripped and feet are symmetrically "
                    "planted. Asymmetric force application places undue stress "
                    "on the shoulder and elbow joints."
                ),
            },
        ],
        "desde_cache": False,
    },
    "tecnica_correcta": {
        "clase": "tecnica_correcta",
        "riesgo": (
            "La técnica es correcta. No se detectan fallos biomecánicos "
            "significativos en esta ejecución. El riesgo de lesión es bajo "
            "cuando se mantiene esta mecánica de forma consistente."
        ),
        "correccion": [
            "Continúa manteniendo la retracción y depresión escapular durante "
            "todo el movimiento para proteger el hombro.",
            "Conserva el leg drive simétrico: los pies deben presionar el suelo "
            "con fuerza constante para transmitir estabilidad al torso.",
            "Mantén la muñeca neutra (sin extensión excesiva) con el agarre "
            "alineado con el antebrazo en todo el recorrido.",
            "Progresa la carga de forma gradual (≤5% semanal) para preservar "
            "la calidad técnica con pesos más elevados.",
            "Incluye variaciones periodizadas (press inclinado, paused bench, "
            "close-grip) para desarrollar puntos débiles sin comprometer la "
            "técnica base.",
        ],
        "fuentes": [
            {
                "titulo": "How to Bench Press – Stronger By Science",
                "url": "https://www.strongerbyscience.com/how-to-bench/",
                "fragmento": (
                    "Proper bench press technique involves a stable arch, "
                    "retracted scapulae, symmetrical leg drive, and a bar path "
                    "that moves slightly toward the face on the way up."
                ),
            },
            {
                "titulo": "Barbell Bench Press – ExRx.net",
                "url": "https://exrx.net/WeightExercises/PectoralSternal/BBBenchPress",
                "fragmento": (
                    "Correct execution: lower the bar to the lower chest with "
                    "elbows at ~75° or less, maintain scapular retraction, and "
                    "press in a slight arc back to the starting position."
                ),
            },
        ],
        "desde_cache": False,
    },
}

# ---------------------------------------------------------------------------
# Gestión de caché JSON
# ---------------------------------------------------------------------------


def _cargar_cache() -> Dict[str, Any]:
    """Carga el archivo de caché JSON. Retorna dict vacío si no existe o es inválido."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("No se pudo leer la caché '%s': %s", CACHE_FILE, exc)
        return {}


def _guardar_cache(cache: Dict[str, Any]) -> None:
    """Persiste el dict de caché en disco."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("No se pudo guardar la caché '%s': %s", CACHE_FILE, exc)


def _entrada_cache_valida(entrada: Dict[str, Any]) -> bool:
    """Verifica que la entrada de caché no haya expirado (TTL = 24 h)."""
    timestamp = entrada.get("_timestamp", 0)
    edad_horas = (time.time() - timestamp) / 3600
    return edad_horas < CACHE_TTL_HORAS


# ---------------------------------------------------------------------------
# Scraping real
# ---------------------------------------------------------------------------


def _obtener_html(url: str) -> str:
    """Realiza un GET a *url* y retorna el texto HTML. Lanza excepción si falla."""
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_HTTP)
    resp.raise_for_status()
    return resp.text


def _extraer_parrafos_relevantes(
    html: str, keywords: List[str], max_parrafos: int = 5
) -> List[str]:
    """
    Parsea el HTML con BeautifulSoup y retorna párrafos que contengan al menos
    una de las *keywords* (búsqueda insensible a mayúsculas).
    """
    soup = BeautifulSoup(html, "html.parser")
    parrafos_relevantes: List[str] = []

    for tag in soup.find_all(["p", "li", "blockquote"]):
        texto = tag.get_text(separator=" ", strip=True)
        if len(texto) < 40:          # ignorar fragmentos muy cortos
            continue
        texto_lower = texto.lower()
        if any(kw.lower() in texto_lower for kw in keywords):
            parrafos_relevantes.append(texto)
            if len(parrafos_relevantes) >= max_parrafos:
                break

    return parrafos_relevantes


def _obtener_titulo_pagina(html: str) -> str:
    """Extrae el título de la página HTML."""
    soup = BeautifulSoup(html, "html.parser")
    tag_title = soup.find("title")
    if tag_title:
        return tag_title.get_text(strip=True)
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return "Sin título"


def _scrape_fuentes(pred_clase: str) -> List[Dict[str, str]]:
    """
    Intenta scrapear las URLs configuradas para *pred_clase*.
    Retorna lista de dicts {"titulo", "url", "fragmento"}.
    Silencia todas las excepciones y retorna lista vacía si nada funciona.
    """
    urls = URLS_POR_CLASE.get(pred_clase, [])
    keywords = KEYWORDS_POR_CLASE.get(pred_clase, [])
    fuentes_encontradas: List[Dict[str, str]] = []

    for url in urls:
        try:
            html = _obtener_html(url)
            titulo = _obtener_titulo_pagina(html)
            parrafos = _extraer_parrafos_relevantes(html, keywords, max_parrafos=3)
            if parrafos:
                fragmento = " ".join(parrafos[:2])
                # Truncar si el fragmento es muy largo
                if len(fragmento) > 600:
                    fragmento = fragmento[:597] + "…"
                fuentes_encontradas.append(
                    {"titulo": titulo, "url": url, "fragmento": fragmento}
                )
        except Exception as exc:  # noqa: BLE001
            logger.info("Scraping fallido para '%s': %s", url, exc)
            continue

    return fuentes_encontradas


# ---------------------------------------------------------------------------
# Función principal pública
# ---------------------------------------------------------------------------


def obtener_diagnostico_clase(
    pred_clase: str,
    forzar_actualizar: bool = False,
) -> Dict[str, Any]:
    """
    Retorna un diagnóstico biomecánico para la clase de técnica detectada.

    Parameters
    ----------
    pred_clase : str
        Una de las cuatro clases:
        ``"codos_abiertos"``, ``"rebote_en_pecho"``,
        ``"empuje_desbalanceado"`` o ``"tecnica_correcta"``.
    forzar_actualizar : bool
        Si es True, ignora la caché y realiza scraping fresco.

    Returns
    -------
    dict con claves:
        ``"clase"``, ``"riesgo"``, ``"correccion"``, ``"fuentes"``,
        ``"desde_cache"``
    """
    # Normalizar entrada
    pred_clase = pred_clase.strip().lower()

    # Validar que la clase sea conocida
    clases_validas = set(FALLBACK_LOCAL.keys())
    if pred_clase not in clases_validas:
        logger.warning(
            "Clase desconocida '%s'. Clases válidas: %s", pred_clase, clases_validas
        )
        # Retornar un diagnóstico genérico sin crashear
        return {
            "clase": pred_clase,
            "riesgo": f"Clase '{pred_clase}' no reconocida. Verifica el nombre.",
            "correccion": ["Verifica que la clase predicha sea válida."],
            "fuentes": [],
            "desde_cache": False,
        }

    # --- 1. Consultar caché ---
    if not forzar_actualizar:
        cache = _cargar_cache()
        if pred_clase in cache and _entrada_cache_valida(cache[pred_clase]):
            resultado = dict(cache[pred_clase])
            resultado.pop("_timestamp", None)   # campo interno, no exponer
            resultado["desde_cache"] = True
            logger.info("Diagnóstico para '%s' servido desde caché.", pred_clase)
            return resultado

    # --- 2. Intentar scraping en vivo ---
    fuentes_scrapeadas: List[Dict[str, str]] = []
    try:
        fuentes_scrapeadas = _scrape_fuentes(pred_clase)
        logger.info(
            "Scraping completado para '%s': %d fuentes.", pred_clase, len(fuentes_scrapeadas)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error inesperado en scraping de '%s': %s", pred_clase, exc)

    # --- 3. Construir resultado fusionando fallback + fuentes scrapeadas ---
    fallback = FALLBACK_LOCAL[pred_clase]

    # Si el scraping obtuvo fuentes, las usa; si no, usa las del fallback
    fuentes_finales = fuentes_scrapeadas if fuentes_scrapeadas else fallback["fuentes"]

    resultado = {
        "clase": pred_clase,
        "riesgo": fallback["riesgo"],
        "correccion": fallback["correccion"],
        "fuentes": fuentes_finales,
        "desde_cache": False,
    }

    # --- 4. Persistir en caché ---
    try:
        cache = _cargar_cache()
        entrada_cache = dict(resultado)
        entrada_cache["_timestamp"] = time.time()
        cache[pred_clase] = entrada_cache
        _guardar_cache(cache)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo actualizar la caché: %s", exc)

    return resultado


# ---------------------------------------------------------------------------
# Utilidades adicionales
# ---------------------------------------------------------------------------


def listar_clases_disponibles() -> List[str]:
    """Retorna las clases con diagnóstico disponible en el fallback local."""
    return list(FALLBACK_LOCAL.keys())


def invalidar_cache_clase(pred_clase: str) -> bool:
    """
    Elimina la entrada de *pred_clase* de la caché JSON.
    Retorna True si se eliminó, False si no existía.
    """
    cache = _cargar_cache()
    if pred_clase in cache:
        del cache[pred_clase]
        _guardar_cache(cache)
        logger.info("Caché invalidada para '%s'.", pred_clase)
        return True
    return False


def invalidar_cache_completa() -> None:
    """Elimina todas las entradas de la caché JSON."""
    _guardar_cache({})
    logger.info("Caché completa invalidada.")


def info_cache() -> Dict[str, Any]:
    """
    Retorna información sobre el estado actual de la caché:
    qué clases hay, edad en horas y si cada una es válida.
    """
    cache = _cargar_cache()
    ahora = time.time()
    info: Dict[str, Any] = {}
    for clase, datos in cache.items():
        if clase.startswith("_"):
            continue
        ts = datos.get("_timestamp", 0)
        edad_h = (ahora - ts) / 3600
        info[clase] = {
            "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "edad_horas": round(edad_h, 2),
            "valida": edad_h < CACHE_TTL_HORAS,
        }
    return info


# ---------------------------------------------------------------------------
# Bloque de prueba rápida (ejecución directa)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    logging.basicConfig(level=logging.INFO, format="%(levelname)s – %(message)s")

    print("=" * 60)
    print("PRUEBA DEL SCRAPER BIOMECÁNICO")
    print("=" * 60)

    for clase in listar_clases_disponibles():
        print(f"\n{'─' * 50}")
        print(f"  Clase: {clase}")
        print(f"{'─' * 50}")
        diag = obtener_diagnostico_clase(clase)
        print(f"  Riesgo   : {diag['riesgo'][:120]}…")
        print(f"  Pasos    : {len(diag['correccion'])}")
        print(f"  Fuentes  : {len(diag['fuentes'])}")
        print(f"  Caché    : {diag['desde_cache']}")

    print("\n\nEstado de caché:")
    pprint.pprint(info_cache())
