import argparse
import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import serial
from ultralytics import YOLO

import app_dashboard as ad


def parse_esp_line(line: str):
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 7:
        return None
    try:
        ax = int(parts[0])
        ay = int(parts[1])
        az = int(parts[2])
        gx = int(parts[3])
        gy = int(parts[4])
        gz = int(parts[5])
        sw = int(parts[6])
        return ax, ay, az, gx, gy, gz, sw
    except ValueError:
        return None


class SerialIMUReader:
    def __init__(self, port: str, baud: int, maxlen: int = 400):
        self.port = port
        self.baud = baud
        self.samples = deque(maxlen=maxlen)
        self.recorded_rows = []
        self.recording = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.status = f"Conectando a {port}..."
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _run(self):
        try:
            with serial.Serial(self.port, self.baud, timeout=0.1) as ser:
                with self.lock:
                    self.status = f"OK · {self.port} @ {self.baud}"
                while not self.stop_event.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    text = raw.decode("utf-8", errors="ignore").strip()
                    if not text or text.startswith("READY_") or text.startswith("ERR_"):
                        continue
                    parsed = parse_esp_line(text)
                    if parsed is None:
                        continue
                    ax, ay, az, gx, gy, gz, sw = parsed
                    t_now = time.monotonic()
                    row = (t_now, ax, ay, az, gx, gy, gz, sw)
                    with self.lock:
                        self.samples.append(row)
                        is_rec = bool(sw == 1)
                        # Si inicia nueva pulsacion tras haber soltado, limpia buffer de la toma anterior
                        if is_rec and not self.recording:
                            self.recorded_rows.clear()
                        self.recording = is_rec
                        if is_rec:
                            self.recorded_rows.append(row)
        except Exception as exc:
            with self.lock:
                self.status = f"Error serial: {exc}"

    def snapshot(self):
        with self.lock:
            return list(self.samples), self.recording, self.status

    def pop_recorded_df(self) -> pd.DataFrame:
        with self.lock:
            rows = list(self.recorded_rows)
            self.recorded_rows.clear()
        if not rows:
            return pd.DataFrame(columns=["t_rel", "ax", "ay", "az", "gx", "gy", "gz", "sw"])
        t0 = rows[0][0]
        return pd.DataFrame(
            [
                {
                    "t_rel": round(r[0] - t0, 4),
                    "ax": r[1],
                    "ay": r[2],
                    "az": r[3],
                    "gx": r[4],
                    "gy": r[5],
                    "gz": r[6],
                    "sw": r[7],
                }
                for r in rows
            ]
        )


def guardar_csv_toma(imu_reader: SerialIMUReader, out_dir: Path, stamp: str) -> Path:
    df_rec = imu_reader.pop_recorded_df()
    csv_out = out_dir / f"vivo_{stamp}.csv"
    df_rec.to_csv(csv_out, index=False)
    print(f"CSV guardado ({len(df_rec)} filas): {csv_out}")
    return csv_out


def calcular_angulo_3pts(a, b, c) -> float:
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return float("nan")
    cos_ang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_ang)))


def extraer_pose_ventana(pose_records: list[dict], feature_names: list[str]) -> dict:
    pose_cols = [c for c in feature_names if c.startswith("pose_")]
    feats = {c: 0.0 for c in pose_cols}
    if not pose_records:
        return feats

    df_p = pd.DataFrame(pose_records)
    for base_col in [c for c in df_p.columns if c != "t"]:
        vals = pd.to_numeric(df_p[base_col], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue
        stats_map = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "rango": float(np.max(vals) - np.min(vals)),
        }
        for stat_name, stat_val in stats_map.items():
            candidate = f"pose_{base_col}_{stat_name}"
            if candidate in feats:
                feats[candidate] = stat_val
    return feats


def extraer_imu_seguro(df_win: pd.DataFrame, fs: float) -> dict:
    if hasattr(ad, "extraer_features_imu"):
        out = ad.extraer_features_imu(df_win, fs)
        if isinstance(out, dict):
            return out
        if isinstance(out, pd.DataFrame) and len(out) > 0:
            return out.iloc[0].to_dict()
    return {}


def dibujar_panel_imu(width: int, height: int, samples: list, fs: float) -> np.ndarray:
    canvas = np.full((height, width, 3), (24, 18, 15), dtype=np.uint8)
    cv2.putText(
        canvas,
        "SENAL IMU EN VIVO (MPU6050)",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
    )

    if len(samples) < 2:
        cv2.putText(
            canvas,
            "Esperando muestras por COM3...",
            (16, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (160, 160, 160),
            1,
        )
        return canvas

    arr = np.array([[s[1], s[2], s[3], s[4], s[5], s[6]] for s in samples], dtype=float)
    norm = (arr - np.mean(arr, axis=0)) / (np.std(arr, axis=0) + 1e-6)
    norm = np.clip(norm, -3.0, 3.0)

    colors = [
        (80, 220, 255),   # ax
        (80, 255, 140),   # ay
        (255, 160, 80),   # az
        (220, 120, 255),  # gx
        (120, 220, 255),  # gy
        (255, 255, 120),  # gz
    ]
    labels = ["ax", "ay", "az", "gx", "gy", "gz"]

    top_y = 55
    bot_y = height - 45
    plot_h = max(60, bot_y - top_y)
    mid_y = top_y + plot_h // 2
    n = len(norm)

    cv2.line(canvas, (16, mid_y), (width - 16, mid_y), (60, 60, 60), 1)

    for ch in range(6):
        pts = []
        for i in range(n):
            x = int(16 + (i / max(1, n - 1)) * (width - 32))
            y = int(mid_y - (norm[i, ch] / 3.0) * (plot_h * 0.45))
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], colors[ch], 1)

        lx = 16 + ch * 68
        cv2.putText(
            canvas,
            f"{labels[ch]}:{int(arr[-1, ch])}",
            (lx, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colors[ch],
            1,
        )

    cv2.putText(
        canvas,
        f"Muestras en buffer: {n} (~{fs:.0f} Hz)",
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (190, 190, 190),
        1,
    )
    return canvas


def predecir_ventana(estimador, feature_names: list[str], combined: dict):
    res = ad.predecir(estimador, feature_names, combined)
    if isinstance(res, dict):
        pred_label = str(res.get("pred", "desconocido"))
        prob = float(res.get("prob_max", 0.0) or 0.0)
        return pred_label, prob
    if isinstance(res, tuple):
        pred_label = str(res[0])
        prob = float(res[1]) if len(res) > 1 and res[1] is not None else 0.0
        return pred_label, prob
    return str(res), 0.0


def main():
    parser = argparse.ArgumentParser(description="Lectura en vivo MPU6050 + YOLOv8-pose")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--cam", type=int, default=2)
    parser.add_argument("--fs", type=float, default=50.0)
    parser.add_argument("--ventana-seg", type=float, default=4.0)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    modelo_path = base_dir / "modelo_press_banca_rf.joblib"
    pose_path = base_dir / "yolov8n-pose.pt"
    out_dir = base_dir / "capturas_vivo"
    out_dir.mkdir(exist_ok=True)

    if not modelo_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo en {modelo_path}")
    if not pose_path.exists():
        raise FileNotFoundError(f"No se encontro yolov8n-pose.pt en {pose_path}")

    bundle = joblib.load(modelo_path)
    estimador, feature_names, _ = ad.desempaquetar_modelo(bundle, base_dir)
    pose_model = YOLO(str(pose_path))

    max_buf = int(args.fs * max(args.ventana_seg * 2, 8.0))
    imu_reader = SerialIMUReader(args.port, args.baud, maxlen=max_buf)
    imu_reader.start()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        imu_reader.stop()
        raise RuntimeError(f"No se pudo abrir la camara con indice {args.cam}")

    pose_history = deque(maxlen=int(30 * max(args.ventana_seg, 5.0)))
    last_pred_text = "ORIENTATIVO: esperando datos..."
    last_pred_time = 0.0
    frame_idx = 0
    last_annotated = None
    video_writer = None
    active_stamp = None
    last_stamp = None
    saving_video_path = None
    gui_available = True
    win_title = "Press Banca - En Vivo (MPU6050 + YOLOv8-pose)"

    print("Controles: [s] guardar CSV+MP4 | [q] o [ESC] salir")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            now_mono = time.monotonic()
            samples, sw_active, serial_status = imu_reader.snapshot()

            if frame_idx % 2 == 0 or last_annotated is None:
                results = pose_model(frame, conf=0.5, verbose=False)
                annotated = results[0].plot() if results else frame.copy()

                ang_rec = {"t": now_mono}
                if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                    kps = results[0].keypoints.xy[0].cpu().numpy()
                    if len(kps) >= 11:
                        ang_codo_izq = calcular_angulo_3pts(kps[5], kps[7], kps[9])
                        ang_codo_der = calcular_angulo_3pts(kps[6], kps[8], kps[10])
                        ang_rec["ang_codo_izq"] = ang_codo_izq
                        ang_rec["ang_codo_der"] = ang_codo_der
                        if not (math.isnan(ang_codo_izq) or math.isnan(ang_codo_der)):
                            ang_rec["asim_codos"] = abs(ang_codo_izq - ang_codo_der)
                pose_history.append(ang_rec)
                last_annotated = annotated
            else:
                annotated = last_annotated.copy()

            frame_idx += 1

            win_samples = int(args.fs * args.ventana_seg)
            if len(samples) >= max(20, win_samples // 2) and (now_mono - last_pred_time) >= 0.6:
                last_pred_time = now_mono
                recent = samples[-win_samples:]
                df_win = pd.DataFrame(
                    [
                        {
                            "ax": s[1],
                            "ay": s[2],
                            "az": s[3],
                            "gx": s[4],
                            "gy": s[5],
                            "gz": s[6],
                        }
                        for s in recent
                    ]
                )
                feats_imu = extraer_imu_seguro(df_win, args.fs)
                recent_pose = [p for p in pose_history if (now_mono - p["t"]) <= args.ventana_seg]
                feats_pose = extraer_pose_ventana(recent_pose, feature_names)

                combined = {}
                for fname in feature_names:
                    if fname in feats_imu:
                        combined[fname] = feats_imu[fname]
                    elif fname in feats_pose:
                        combined[fname] = feats_pose[fname]
                    else:
                        combined[fname] = 0.0

                pred_label, prob = predecir_ventana(estimador, feature_names, combined)
                last_pred_text = f"ORIENTATIVO ({args.ventana_seg:.0f}s): {pred_label} ({prob * 100:.1f}%)"

            sw_tag = "BOTON: GRABANDO (sw=1)" if sw_active else "BOTON: SUELTO (sw=0)"
            sw_color = (60, 220, 80) if sw_active else (180, 180, 180)
            cv2.rectangle(annotated, (10, 10), (annotated.shape[1] - 10, 72), (18, 18, 22), -1)
            cv2.putText(
                annotated,
                last_pred_text,
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (80, 235, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"{sw_tag} | {serial_status}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                sw_color,
                1,
            )

            if sw_active:
                if video_writer is None:
                    active_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    last_stamp = active_stamp
                    saving_video_path = out_dir / f"vivo_{active_stamp}.mp4"
                    h_f, w_f = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(saving_video_path), fourcc, 20.0, (w_f, h_f))
                video_writer.write(annotated)
            elif video_writer is not None:
                video_writer.release()
                video_writer = None
                print(f"Grabacion detenida al soltar boton. Video: {saving_video_path}")
                if active_stamp:
                    guardar_csv_toma(imu_reader, out_dir, active_stamp)
                    active_stamp = None

            imu_panel = dibujar_panel_imu(460, annotated.shape[0], list(samples)[-win_samples:], args.fs)
            combined_view = np.hstack((annotated, imu_panel))

            if gui_available:
                try:
                    cv2.imshow(win_title, combined_view)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("s"):
                        stamp_s = active_stamp or last_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
                        guardar_csv_toma(imu_reader, out_dir, stamp_s)
                        if saving_video_path and saving_video_path.exists():
                            print(f"Video disponible para cargar en dashboard: {saving_video_path}")
                except cv2.error:
                    gui_available = False
                    print("Aviso: OpenCV sin soporte de ventana GUI (headless). Presiona Ctrl+C en consola para salir.")
            else:
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nDetenido por el usuario (Ctrl+C).")
    finally:
        if video_writer is not None:
            video_writer.release()
            if active_stamp:
                guardar_csv_toma(imu_reader, out_dir, active_stamp)
        cap.release()
        imu_reader.stop()
        if gui_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()