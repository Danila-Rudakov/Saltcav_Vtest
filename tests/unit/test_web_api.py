"""Тесты API локального сервера визуализации saltcav.web."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import urllib.error
from http import HTTPStatus
from http.server import ThreadingHTTPServer

import pytest

from saltcav.web.server import SaltcavRequestHandler, get_default_config_dict


@pytest.fixture(scope="module")
def live_server():
    """Запустить тестовый HTTP-сервер на свободном порту."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), SaltcavRequestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def test_get_defaults(live_server: str) -> None:
    """GET /api/defaults возвращает валидную конфигурацию."""
    url = f"{live_server}/api/defaults"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == HTTPStatus.OK
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "ok"
        cfg = data["config"]
        assert cfg["setup"] in ("channel", "window")
        assert cfg["length_mm"] > 0
        assert len(cfg["stages"]) >= 1


def test_post_simulate_success(live_server: str) -> None:
    """POST /api/simulate рассчитывает каверну и возвращает геометрию."""
    payload = {
        "name": "test-web-sim",
        "setup": "channel",
        "model": "section",
        "natural_convection": True,
        "length_mm": 50.0,
        "diameter_mm": 2.5,
        "max_radius_mm": 20.0,
        "temperature_c": 20.0,
        "salt_density_kg_m3": 2160.0,
        "roof_mult": 1.0,
        "wall_mult": 1.0,
        "sh_mult": 1.0,
        "n_cells": 10,
        "n_theta": 6,
        "output_every_min": 15.0,
        "stages": [
            {"duration_h": 0.25, "flow_ml_min": 10.0, "inlet_saturation": 0.0}
        ],
    }

    url = f"{live_server}/api/simulate"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == HTTPStatus.OK
        res = json.loads(resp.read().decode("utf-8"))
        assert res["status"] == "ok"
        d = res["data"]
        assert d["config"]["name"] == payload["name"]
        assert d["config"]["stages"] == payload["stages"]
        assert d["config"]["depth_m"] == 0
        assert d["n_steps"] > 0
        assert len(d["times"]) == d["n_steps"]
        assert len(d["x_mm"]) == 11  # n_cells + 1
        assert len(d["theta_rad"]) == 7  # n_theta + 1
        assert len(d["radius_mm"]) == d["n_steps"]
        assert "metrics" in d
        assert len(d["metrics"]["volume_cm3"]) == d["n_steps"]


def test_post_simulate_full_contour_with_material(live_server: str) -> None:
    """Режим full: замкнутый контур 0…2π, материал и условие Стефана доходят до модели."""
    payload = {
        "name": "test-web-full",
        "setup": "full",
        "model": "section",
        "natural_convection": True,
        "length_mm": 40.0,
        "diameter_mm": 4.0,
        "max_radius_mm": 20.0,
        "n_cells": 8,
        "n_theta": 12,
        "output_every_min": 15.0,
        "stefan_condition": True,
        "material": {
            "seed": 1,
            "grain_contrast": 0.2,
            "boundary_factor": 1.65,
            "sheet_offset_mm": 25.0,
        },
        "stages": [{"duration_h": 0.25, "flow_ml_min": 1.0, "inlet_saturation": 0.0}],
    }
    req = urllib.request.Request(
        f"{live_server}/api/simulate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read().decode("utf-8"))["data"]
    assert len(d["theta_rad"]) == 13
    assert d["theta_rad"][-1] == pytest.approx(2 * 3.141592653589793, abs=1e-3)
    assert d["halves"] == 1
    assert d["config"]["material"]["seed"] == 1


def test_post_simulate_with_tube_pockets(live_server: str) -> None:
    """Заход трубок: ось x уходит в отрицательные значения, расчётная область расширена."""
    payload = {
        "name": "test-web-tubes",
        "setup": "full",
        "model": "section",
        "length_mm": 40.0,
        "diameter_mm": 4.0,
        "max_radius_mm": 20.0,
        "n_cells": 16,
        "n_theta": 12,
        "output_every_min": 15.0,
        "tube_insert_mm": 5.0,
        "tube_gap_mm": 0.1,
        "pocket_exchange_mult": 10.0,
        "stages": [{"duration_h": 0.25, "flow_ml_min": 1.0, "inlet_saturation": 0.0}],
    }
    req = urllib.request.Request(
        f"{live_server}/api/simulate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read().decode("utf-8"))["data"]
    assert d["n_pocket"] == 2
    assert d["x_mm"][0] == pytest.approx(-5.0)
    assert d["x_mm"][-1] == pytest.approx(45.0)
    assert len(d["radius_mm"][-1]) == len(d["x_mm"])


def test_post_simulate_material_requires_full(live_server: str) -> None:
    """material при setup: channel — ошибка валидации (HTTP 400)."""
    payload = {
        "name": "bad-material",
        "setup": "channel",
        "length_mm": 40.0,
        "diameter_mm": 4.0,
        "material": {"seed": 1},
        "stages": [{"duration_h": 0.1, "flow_ml_min": 1.0}],
    }
    req = urllib.request.Request(
        f"{live_server}/api/simulate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == HTTPStatus.BAD_REQUEST


def test_post_simulate_invalid_config(live_server: str) -> None:
    """POST /api/simulate с некорректными данными возвращает HTTP 400."""
    payload = {
        "name": "invalid",
        "length_mm": -10.0,  # Отрицательная длина недопустима
        "diameter_mm": 2.5,
        "stages": [],
    }

    url = f"{live_server}/api/simulate"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == HTTPStatus.BAD_REQUEST


def test_static_files_served(live_server: str) -> None:
    """Статические файлы index.html, style.css, app.js успешно отдаются."""
    for path in ["/", "/index.html", "/style.css", "/app.js", "/report.js"]:
        url = f"{live_server}{path}"
        with urllib.request.urlopen(url) as resp:
            assert resp.status == HTTPStatus.OK
            content = resp.read()
            assert len(content) > 0
