"""Tests for the dashboard's look-and-feel helpers (no Streamlit needed)."""
import re
from pathlib import Path

import pytest

from smart_listener import dashboard_ui as ui

ROOT = Path(__file__).resolve().parents[1]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐⭕⏩-⏺]")


def test_incident_codes_and_machine_types():
    assert ui.inc_code(1) == "INC-001" and ui.inc_code(42) == "INC-042" and ui.inc_code(1234) == "INC-1234" and ui.inc_code(None) == "-"
    assert ui.machine_type_of("pump_id_04") == "Pump" and ui.machine_type_of("fan_id_00") == "Fan"


def test_times_are_shown_in_12_hour_utc():
    assert ui.fmt_time("2026-11-09T10:30:00+00:00") == "10:30 AM, 09 Nov 2026"
    assert ui.fmt_time("2026-11-09T22:05:00+00:00", date=False) == "10:05 PM"
    assert ui.fmt_time("2026-11-09T15:00:00+05:00") == "10:00 AM, 09 Nov 2026"      # converted to UTC
    assert ui.fmt_time("2026-11-09T10:30:00") == "10:30 AM, 09 Nov 2026"            # no zone = UTC
    assert ui.fmt_time(None) == "" and ui.fmt_time("garbage") == ""


def test_severity_colours_follow_the_brief():
    assert ui.SEVERITY["NORMAL"].lower() in ("#3fb27f",) and ui.SEVERITY["HIGH"].startswith("#e5")
    amber = lambda c: int(c[1:3], 16) > 0xb0 and 0x70 < int(c[3:5], 16) < 0xc0 and int(c[5:7], 16) < 0x60
    assert amber(ui.SEVERITY["WARNING"]) and amber(ui.SEVERITY["WATCH"])


def test_branding_machine_codes_and_inter_font():
    assert ui.PRODUCT_FULL == "Smart Listener - Industrial Acoustic Anomaly Detection"
    assert ui.machine_code("pump_id_04") == "PUMP-04" and ui.machine_code("fan_id_00") == "FAN-00"
    assert "family=Inter" in ui.CSS and '"Inter"' in ui.CSS
    assert "Smart Listener" in ui.appbar() and "SIMULATED" in ui.appbar('<span class="sl-chip">SIMULATED MONITORING</span>')
    assert set(ui.STATE_TEXT) == set(ui.SEVERITY)


def test_table_escapes_nothing_it_is_told_is_trusted_but_helpers_escape_text():
    assert ui.esc("<script>") == "&lt;script&gt;" and ui.esc(None) == "" and ui.esc('"x"') == "&quot;x&quot;"
    t = ui.table(["A<b>"], [[ui.esc("<i>")]])
    assert "<b>" not in t.replace("<th", "").split("A")[1][:3] and "&lt;i&gt;" in t
    assert "Nothing" in ui.table(["a"], [])


def test_pills_and_banner():
    assert "HIGH" in ui.pill("HIGH") and "NO DATA" in ui.pill("NO_DATA") and "ABOVE ALARM LIMIT" in ui.clip_pill("above_alarm_limit")
    assert "sl-banner high" in ui.banner("high", "x") and ui.source_tag("simulated").count("SIMULATED") == 1 and ui.source_tag(None) == ""


def test_no_emojis_or_placeholder_labels_in_the_interface_code():
    app = (ROOT / "dashboard/app.py").read_text(encoding="utf-8")
    for bad in ("DEMO ENVIRONMENT", "ILLUSTRATIVE", "Prototype screen", "MIMII DATASET", "SAMPLE", "lorem"):
        assert bad not in app, bad
    for f in ("dashboard/app.py", "src/smart_listener/dashboard_ui.py", "src/smart_listener/evidence.py",
              "src/smart_listener/assistant.py", "src/smart_listener/reports.py"):
        found = EMOJI.findall((ROOT / f).read_text(encoding="utf-8"))
        assert not found, (f, found)


def test_theme_file_is_dark_navy_with_steel_accent():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text()
    assert 'base = "dark"' in cfg and "#121b27" in cfg


def test_no_dependency_beyond_streamlit_was_added():
    reqs = [l.strip() for l in (ROOT / "requirements-app.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert reqs == ["fastapi", "uvicorn", "httpx", "python-multipart", "streamlit>=1.50"]
