# Detección y Clasificación Automática de Errores Técnicos en Press de Banca con Barra

[![Abrir en Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://press-banca-dashboard-nn3haqwzmqllkybbtuy65b.streamlit.app/)

🔗 **Demo en vivo:** https://press-banca-dashboard-nn3haqwzmqllkybbtuy65b.streamlit.app/

Proyecto integrador desarrollado en el programa de **Ingeniería de Datos e Inteligencia Artificial** (**Universidad Santo Tomás**, Tunja) por **Cristian Leonardo Barrero Zambrano**, **Samuel Tobías Dueñas Guerrero**, **Juan Manuel Rubio Baez** y **Jean Pier Felipe Rincón**.

El sistema combina captura inercial de 6 ejes (`ESP32 DevKit` + `MPU6050` vía bus I2C en `0x68` a `50.0 Hz` por puerto serie a `115200` baudios) con análisis de pose mediante `YOLOv8n-pose` (`yolov8n-pose.pt`), extracción de descriptores temporales y espectrales (`FFT` sin componente DC), clasificación con **Random Forest** (200 árboles) y un tablero interactivo en **Streamlit** apoyado en `scraper_biomecanica.py`.

## Resultados principales (prueba `n = 16`)

- **Exactitud global:** `81.25 %` (`13/16` aciertos)
- **F1 macro:** `0.817` (precisión macro `0.850`, recall macro `0.812`)
- **Dataset base:** `160` pares sincronizados (`144` entrenamiento: 36 por clase; `16` prueba: 4 por clase) en `df_full_fusion.csv`
- **Clases evaluadas:** `tecnica_correcta`, `codos_abiertos`, `rebote_en_pecho` y `empuje_desbalanceado`

## Estructura del repositorio

- `panel_de_aplicación.py` *(o `app_dashboard.py` según el entrypoint vigente)*: Interfaz principal en **Streamlit** para explorar muestras (`train`/`test`), inspeccionar señales IMU y espectros `FFT`, reproducir videos en `MP4 H.264` (`yuv420p`), consultar importancia de variables y ver el diagnóstico biomecánico.
- `en_vivo.py` e `integracion_vivo.py`: Captura sincronizada en tiempo real desde puerto serie (por ejemplo `COM3` a `115200` baudios) y cámara, guardando sesiones en `capturas_vivo/`.
- `scraper_biomecanica.py` y `base_conocimiento_press.json`: Módulo de *web scraping* y consulta biomecánica indexado por clase técnica para mostrar riesgos y pautas de corrección.
- `modelo_press_banca_rf.joblib` y `df_full_fusion.csv`: Modelo Random Forest entrenado y tabla consolidada de características multimodales.
- `generar_videos_test.py` y `renombrar_pares.py`: Utilidades para exportar videos anotados compatibles con navegador y organizar pares de archivos.
- `capturas_vivo/`, `pruebas_modelo/` y carpetas por clase (`tecnica_correcta/`, `codos_abiertos/`, `rebote_en_pecho/`, `empuje_desbalanceado/`): Datos de prueba, grabaciones y muestras del proyecto.

## Ejecución en local

La captura en vivo con puerto serie y cámara debe ejecutarse en local (`http://localhost:8501`), ya que el entorno en la nube no dispone de acceso directo al puerto serie ni a la cámara física del equipo:

```bash
pip install -r requisitos.txt
streamlit run panel_de_aplicación.py
