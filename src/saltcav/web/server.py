"""Локальный HTTP-сервер для веб-приложения визуализации каверны (saltcav).

Предоставляет API:
- GET /api/defaults — параметры по умолчанию из configs/lab/channel.yaml
- POST /api/simulate — запуск расчета simulate_section и возврат кадров анимации
- Статика — отдача HTML, CSS, JS из папки static/
"""

from __future__ import annotations

import json
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from saltcav.core.lab_config import LabConfig, load_lab_config
from saltcav.leaching.section import simulate_section

STATIC_DIR = Path(__file__).resolve().parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_BODY_SIZE = 5 * 1024 * 1024  # 5 МБ предел размера тела запроса


def get_default_config_dict() -> dict[str, Any]:
    """Загрузить дефолтный конфиг channel.yaml или вернуть базовый словарь."""
    yaml_candidates = [
        PROJECT_ROOT / "configs" / "lab" / "channel.yaml",
        Path.cwd() / "configs" / "lab" / "channel.yaml",
        Path.cwd() / "Saltcav_Vtest" / "configs" / "lab" / "channel.yaml",
    ]
    for p in yaml_candidates:
        if p.exists():
            try:
                cfg = load_lab_config(p)
                return cfg.model_dump()
            except Exception:
                pass

    # Дефолт при отсутствии файла
    return {
        "name": "lab-channel-example",
        "setup": "channel",
        "model": "section",
        "natural_convection": True,
        "length_mm": 200.0,
        "diameter_mm": 2.5,
        "max_radius_mm": 40.0,
        "temperature_c": 20.0,
        "salt_density_kg_m3": 2160.0,
        "sh_mult": 1.0,
        "roof_mult": 1.0,
        "wall_mult": 1.0,
        "n_cells": 80,
        "n_theta": 18,
        "output_every_min": 2.0,
        "stages": [
            {"duration_h": 1.0, "flow_ml_min": 5.0, "inlet_saturation": 0.0},
            {"duration_h": 1.0, "flow_ml_min": 20.0, "inlet_saturation": 0.0},
        ],
    }


def format_seconds(seconds: float) -> str:
    """Форматирование секунд в строку вида '01ч 15м'."""
    total_minutes = int(round(seconds / 60.0))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}ч {minutes:02d}м"


class SaltcavRequestHandler(SimpleHTTPRequestHandler):
    """HTTP-обработчик с раздачей статики и API-эндпоинтами."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self) -> None:
        # Разрешаем CORS и отключаем кэш для свежих данных API
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path == "/api/defaults":
            self.handle_get_defaults()
            return

        if path == "/" or path == "":
            self.path = "/index.html"

        return super().do_GET()

    def do_POST(self) -> None:
        path = self.path.split("?")[0]

        if path == "/api/simulate":
            self.handle_post_simulate()
            return

        self.send_error(HTTPStatus.NOT_FOUND, f"Endpoint {path} not found")

    def handle_get_defaults(self) -> None:
        defaults = get_default_config_dict()
        self.send_json_response(HTTPStatus.OK, {"status": "ok", "config": defaults})

    def handle_post_simulate(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self.send_json_response(
                HTTPStatus.BAD_REQUEST, {"status": "error", "error": "Пустое тело запроса"}
            )
            return

        if content_length > MAX_BODY_SIZE:
            self.send_json_response(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"status": "error", "error": "Размер запроса превышает 5 МБ"},
            )
            return

        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            self.send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": f"Ошибка разбора JSON: {e}"},
            )
            return

        # Приводим типы и валидируем через LabConfig
        try:
            cfg = LabConfig.model_validate(payload)
        except Exception as e:
            self.send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": f"Ошибка конфигурации: {e}"},
            )
            return

        # Запуск расчета
        t0 = time.time()
        try:
            res = simulate_section(cfg)
        except Exception as e:
            self.send_json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"status": "error", "error": f"Ошибка расчета: {e}"},
            )
            return
        calc_duration = time.time() - t0

        # Сериализация результатов в компактный формат (единицы в мм для удобства отрисовки)
        x_mm = [round(float(val * 1e3), 2) for val in res.x]
        theta_rad = [round(float(val), 4) for val in res.theta]
        times_s = [round(float(t), 1) for t in res.times]
        formatted_times = [format_seconds(t) for t in res.times]

        # Радиусы: [шаг_времени][узел_x][угол_theta] в мм
        radius_mm = [
            [[round(float(r * 1e3), 3) for r in x_slice] for x_slice in step]
            for step in res.radius
        ]

        # Метрики
        metrics = {
            "volume_cm3": [round(float(v * 1e6), 3) for v in res.volume],
            "outlet_concentration_g_l": [
                round(float(c), 2) for c in res.outlet_concentration
            ],
            "c_sat_g_l": round(float(res.c_sat), 2),
            "flow_ml_min": [round(float(q * 1e6 * 60.0), 2) for q in res.flow],
            "dissolved_g": [round(float(m * 1e3), 2) for m in res.dissolved],
            "calc_duration_s": round(calc_duration, 3),
        }

        response_data = {
            "status": "ok",
            "data": {
                "config": cfg.model_dump(),
                "n_steps": len(times_s),
                "times": times_s,
                "formatted_times": formatted_times,
                "x_mm": x_mm,
                "theta_rad": theta_rad,
                "radius_mm": radius_mm,
                "initial_diameter_mm": float(cfg.diameter_mm),
                "max_radius_mm": float(cfg.max_radius_mm) if cfg.max_radius_mm else None,
                "halves": res.halves,
                "metrics": metrics,
            },
        }

        self.send_json_response(HTTPStatus.OK, response_data)

    def send_json_response(self, status: HTTPStatus, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def launch_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Запустить сервер локально."""
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, SaltcavRequestHandler)
    print(f"==================================================")
    print(f"  SALTCAV LAB Web Server запущен!")
    print(f"  Адрес: http://{host}:{port}")
    print(f"  Для остановки нажмите Ctrl+C")
    print(f"==================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка сервера...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    launch_server(port=port)
