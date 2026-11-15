"""Look and feel of the dashboard: colours, CSS, and small HTML builders (no Streamlit import, so it is testable).

Design (from the Figma screens): dark navy/graphite surfaces, steel-blue accents, Inter type, rounded cards.
Severity colours: NORMAL green, WATCH and WARNING amber (WATCH muted, WARNING strong), HIGH red.
Everything that comes from the database is HTML-escaped before it is placed in a page.
"""
from datetime import date as _date, datetime, time as _time, timedelta, timezone
from html import escape
from typing import Optional

# ----------------------------------------------------------------------------- local time (Pakistan)
# Every time is STORED in UTC. It is converted to Pakistan Standard Time only when it is shown (website, reports,
# CSV, e-mails). Windows Python may have no time-zone database (the 'tzdata' package provides it); Pakistan has no
# daylight saving time, so a fixed UTC+05:00 is then exactly the same.
TZ_NAME = "Asia/Karachi"
TZ_LABEL = "PKT"
try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo(TZ_NAME)
except Exception:  # ZoneInfoNotFoundError (no tzdata) or a very old Python
    LOCAL_TZ = timezone(timedelta(hours=5), TZ_LABEL)

PRODUCT = "Smart Listener"
PRODUCT_SUB = "Industrial Acoustic Anomaly Detection"
PRODUCT_FULL = f"{PRODUCT} - {PRODUCT_SUB}"
DESCRIPTION = ("A system that analyses industrial machinery sounds to detect unusual acoustic patterns, assess "
               "operational severity and monitor machine incidents.")

SEVERITY = {  # label colour
    "NORMAL": "#3fb27f", "WATCH": "#d9a23c", "WARNING": "#f0a31c", "HIGH": "#e5534b", "NO_DATA": "#7d8b9b"}
SEVERITY_BG = {"NORMAL": "#1d3a33", "WATCH": "#3b3324", "WARNING": "#403220", "HIGH": "#43262b", "NO_DATA": "#26313e"}
SEVERITY_LABEL = {"NO_DATA": "NO DATA"}
STATE_TEXT = {"NORMAL": "Within normal range", "WATCH": "Isolated deviation", "WARNING": "Repeated deviation",
              "HIGH": "Sustained deviation", "NO_DATA": "Not analysed yet"}
CLIP = {"within_normal": ("WITHIN NORMAL RANGE", "#3fb27f"), "above_warning_limit": ("ABOVE WARNING LIMIT", "#f0a31c"),
        "above_alarm_limit": ("ABOVE ALARM LIMIT", "#e5534b")}
ALERT_STATUS = {   # e-mail delivery status -> (label, colour, background)
    "sent": ("SENT", "#3fb27f", "#1d3a33"), "dry_run": ("TEST MODE - NOT SENT", "#d9a23c", "#3b3324"),
    "pending": ("PENDING", "#79aedb", "#20344a"), "failed": ("FAILED", "#e5534b", "#43262b"),
    "suppressed": ("SUPPRESSED", "#a9b6c4", "#26313e")}
ALERT_KIND = {"high_opened": "HIGH incident opened", "reminder": "Reminder", "test": "Test e-mail", "closed": "Closed",
              "analysis": "Upload analysis result"}
SEVERITY_ORDER = {"HIGH": 0, "WARNING": 1, "WATCH": 2, "NORMAL": 3, "NO_DATA": 4}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#121b27;--bar:#0f1721;--side:#162130;--card:#1b2838;--card2:#152130;--line:#2a3a4e;--text:#e6ecf2;
--muted:#97a6b7;--dim:#6f8093;--steel:#34597c;--steel2:#3f6a92;--link:#79aedb;
--normal:#3fb27f;--watch:#d9a23c;--warning:#f0a31c;--high:#e5534b}
html,body,[class*="css"],[data-testid="stAppViewContainer"],button,input,textarea,select,p,li,label,
[data-testid="stMarkdownContainer"]{font-family:"Inter","Segoe UI","Helvetica Neue",Arial,sans-serif !important}
[data-testid="stAppViewContainer"]{background:var(--bg)}
header[data-testid="stHeader"]{display:none}
[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarHeader"]{display:none}
.block-container{padding-top:92px !important;padding-bottom:3rem;padding-left:2.4rem !important;padding-right:2.4rem !important;max-width:1560px}
/* fixed application bar */
.sl-appbar{position:fixed;top:0;left:0;right:0;height:66px;background:var(--bar);border-bottom:1px solid var(--line);
z-index:999990;display:flex;align-items:center;justify-content:space-between;padding:0 28px}
.sl-appbar .name{font-size:20px;font-weight:600;color:var(--text);letter-spacing:.2px;line-height:1.1}
.sl-appbar .sub{font-size:10.5px;color:var(--muted);letter-spacing:1.3px;text-transform:uppercase;margin-top:4px}
.sl-appbar .right{display:flex;align-items:center;gap:14px;font-size:12.5px;color:var(--muted)}
.sl-chip{display:inline-block;white-space:nowrap;padding:4px 12px;border-radius:14px;font-size:11px;font-weight:600;letter-spacing:.8px;
background:#3d3222;color:var(--warning)}
.sl-chip.steel{background:#20344a;color:var(--link)}
/* sidebar navigation */
[data-testid="stSidebar"]{background:var(--side);border-right:1px solid var(--line);top:66px !important;
height:calc(100vh - 66px) !important;width:236px !important;min-width:236px !important;max-width:236px !important}
[data-testid="stSidebar"] > div{width:236px !important}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding-top:18px}
.sl-nav-label{font-size:11px;color:var(--dim);letter-spacing:1.2px;text-transform:uppercase;margin:10px 4px 6px}
.sl-nav-sep{border-top:1px solid var(--line);margin:18px 0 8px}
[data-testid="stSidebar"] .stButton>button{justify-content:flex-start;text-align:left;border:0;background:transparent;
color:#c3cfdb;font-size:15px;font-weight:500;padding:9px 16px;border-radius:6px;min-height:44px}
[data-testid="stSidebar"] .stButton>button div{justify-content:flex-start}
[data-testid="stSidebar"] .stButton>button:hover{background:#1f2d3f;color:#fff}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:var(--steel);color:#fff}
.sl-user{font-size:12.5px;color:var(--muted);margin:4px 4px 6px}
/* page heading */
.sl-title{font-size:30px;font-weight:500;color:var(--text);margin:0 0 6px;letter-spacing:-.2px}
.sl-sub{font-size:14.5px;color:var(--muted);margin:0 0 22px}
.sl-h{font-size:19px;font-weight:500;color:var(--text);margin:0 0 4px}
.sl-hsub{font-size:13.5px;color:var(--muted);margin:0 0 14px}
.sl-label{font-size:11px;color:var(--muted);letter-spacing:1.1px;text-transform:uppercase;margin:0 0 4px}
.sl-help{font-size:13px;color:var(--muted);line-height:1.55;margin:4px 0 10px}
.sl-dim{color:var(--dim)}
/* cards: st.container(border=True, key="card-...") */
[class*="st-key-card"]{background:var(--card) !important;border:1px solid var(--line) !important;border-radius:10px !important;
padding:22px 24px !important}
[class*="st-key-card"] [class*="st-key-card"]{background:var(--card2) !important}
[class*="st-key-row-"]{border-bottom:1px solid var(--line);padding:4px 0 4px}
[class*="st-key-row-"]:last-child{border-bottom:0}
.sl-th{font-size:11px;color:var(--muted);letter-spacing:1.1px;text-transform:uppercase;padding:0 0 4px}
.sl-cell{font-size:14px;color:var(--text)}
.sl-cell.strong{font-weight:600}
/* KPI */
.sl-kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px;height:100%}
.sl-kpi .k{font-size:11.5px;color:var(--muted);letter-spacing:.9px;text-transform:uppercase}
.sl-kpi .v{font-size:34px;font-weight:500;margin:6px 0 2px;line-height:1.1}
.sl-kpi .d{font-size:13px;color:var(--muted)}
/* stat strip */
.sl-strip{display:flex;border:1px solid var(--line);background:var(--card2);border-radius:8px;margin:0 0 16px;flex-wrap:wrap}
.sl-stat{display:flex;flex-direction:column;justify-content:space-between;flex:1;min-width:130px;padding:12px 16px;border-right:1px solid var(--line)}
.sl-stat:last-child{border-right:0}
.sl-stat:has(.sl-pill){flex:1.6;min-width:190px}
.sl-stat .k{font-size:11px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;line-height:1.4}
.sl-stat .k i{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:7px;vertical-align:1px}
.sl-stat .v{font-size:24px;font-weight:500;color:var(--text);margin-top:4px;line-height:1.15}
.sl-stat .v.sm{font-size:15px;margin-top:6px}
/* pills */
.sl-pill{display:inline-flex;align-items:center;gap:7px;padding:3px 12px;border-radius:14px;font-size:11.5px;
font-weight:600;letter-spacing:.7px;white-space:nowrap}
.sl-pill i{width:6px;height:6px;border-radius:50%;display:inline-block}
.sl-pill.lg{font-size:13px;padding:5px 15px}
.sl-tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:10px;font-weight:600;letter-spacing:.8px;vertical-align:middle}
.sl-tag.sim{color:#c99a45;background:transparent;border:1px solid #57472a;font-size:9px;padding:0 6px}
.sl-tag.up{color:var(--link);background:transparent;border:1px solid #2c4560;font-size:9px;padding:0 6px}
/* banners */
.sl-banner{border-left:3px solid var(--line);background:var(--card2);border-radius:6px;padding:11px 16px;
margin:0 0 12px;font-size:14px;color:var(--text);line-height:1.55}
.sl-banner b{font-weight:600}
.sl-banner.high{border-left-color:var(--high);background:#2f1f26}
.sl-banner.warn,.sl-banner.sim{border-left-color:var(--warning);background:#2c2a24}
.sl-banner.info{border-left-color:var(--link);background:#18283a}
.sl-banner.ok{border-left-color:var(--normal);background:#18302b}
/* key/value grid */
.sl-kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px 22px;margin:4px 0 16px}
.sl-kv .k{font-size:11px;color:var(--muted);letter-spacing:1px;text-transform:uppercase}
.sl-kv .v{font-size:15px;color:var(--text);margin-top:4px;font-weight:500;word-break:break-word}
/* HTML tables (override Streamlit's default markdown table borders) */
.sl-table,.sl-facts{border:0 !important}
.sl-table th,.sl-table td,.sl-facts td{border-left:0 !important;border-right:0 !important;border-top:0 !important;background:transparent}
.sl-table{width:100%;border-collapse:collapse;font-size:13.5px}
.sl-table th{color:var(--muted);font-size:11px;font-weight:500;letter-spacing:1px;text-transform:uppercase;text-align:left;
padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
.sl-table td{padding:10px 10px;border-bottom:1px solid var(--line);color:var(--text);vertical-align:middle}
.sl-table tr:last-child td{border-bottom:0}
.sl-table tr.high td:first-child{box-shadow:inset 3px 0 0 var(--high)}
.sl-table tr.warning td:first-child{box-shadow:inset 3px 0 0 var(--warning)}
.sl-table .num{text-align:right;font-variant-numeric:tabular-nums}
.sl-table .nw,.sl-table .mono{white-space:nowrap}
.sl-table .dim{color:var(--dim)}
.mono{font-variant-numeric:tabular-nums}
.sl-facts{width:100%;border-collapse:collapse;font-size:13.5px}
.sl-facts td{padding:7px 0;border-bottom:1px solid var(--line);color:var(--text)}
.sl-facts td:first-child{color:var(--muted);width:52%}
.sl-facts td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.sl-facts.left td:first-child{width:40%}.sl-facts.left td:last-child{text-align:left}
.sl-bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;min-width:70px}
.sl-bar span{display:block;height:100%;background:var(--link)}
/* steps, activity, chat */
.sl-step{display:flex;gap:14px;margin:0 0 18px}
.sl-step .n{flex:0 0 auto;font-size:11.5px;color:var(--link);background:#20344a;border-radius:10px;padding:3px 10px;height:fit-content}
.sl-step .t{font-size:15px;color:var(--text)}
.sl-step .d{font-size:13px;color:var(--muted);margin-top:3px}
.sl-act{display:flex;gap:16px;padding:11px 0;border-bottom:1px solid var(--line);font-size:13.5px;color:var(--text)}
.sl-act:last-child{border-bottom:0}
.sl-act .t{color:var(--dim);flex:0 0 64px;font-variant-numeric:tabular-nums}
.sl-q{background:#22344a;border-radius:8px;padding:14px 18px;margin:6px 0 16px auto;max-width:75%;font-size:14.5px;color:var(--text)}
.sl-a{background:var(--card2);border-radius:8px;padding:6px 20px 4px;margin:0 0 18px}
.sl-a .who{font-size:11px;color:var(--link);letter-spacing:1px;text-transform:uppercase;margin:6px 0 2px}
.sl-a .sec{padding:12px 0;border-bottom:1px solid var(--line)}
.sl-a .sec:last-child{border-bottom:0}
.sl-a .lab{font-size:11px;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px}
.sl-a .txt{font-size:14.5px;color:var(--text);line-height:1.55}
/* widgets */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{border-radius:6px;border:1px solid var(--line);
background:#22324a;color:var(--text);font-weight:500;font-size:14px;min-height:42px;padding:0 18px}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{border-color:var(--steel2);color:#fff}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primaryFormSubmit"],.stDownloadButton>button[kind="primary"]{
background:var(--steel);border-color:var(--steel);color:#fff}
.stButton>button[kind="primary"]:hover,.stFormSubmitButton>button[kind="primaryFormSubmit"]:hover{background:var(--steel2)}
.stButton>button p,.stFormSubmitButton>button p,.stDownloadButton>button p{white-space:nowrap}
.stButton>button:disabled,.stFormSubmitButton>button:disabled{opacity:.4;cursor:not-allowed}
.stButton>button[kind="tertiary"]{border:0;background:transparent;color:var(--link);min-height:30px;padding:0;font-size:13.5px}
.stButton>button[kind="tertiary"]:hover{color:#a9cdf0;text-decoration:underline}
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="textarea"],[data-testid="stFileUploaderDropzone"]{
background:var(--card2) !important;border-radius:6px !important}
[data-testid="stFileUploaderDropzone"]{border:1px dashed #3a4d63 !important;padding:26px !important}
[data-testid="stWidgetLabel"] p{font-size:13.5px;color:#c3cfdb}
[data-testid="stTabs"] button p{font-size:14px;font-weight:500}
[data-testid="stImage"] img{border-radius:6px}
div[data-testid="stExpander"] details{border:1px solid var(--line);border-radius:8px;background:var(--card2)}
[data-testid="stForm"]{border:0;padding:0}
</style>
"""


# ----------------------------------------------------------------------------- small helpers
def esc(x) -> str:
    return escape("" if x is None else str(x), quote=True)


def inc_code(incident_id) -> str:
    return f"INC-{int(incident_id):03d}" if incident_id else "-"


def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def to_local(ts) -> Optional[datetime]:
    """A stored UTC time (text or datetime) as an aware datetime in Pakistan time. None if there is no time."""
    d = parse_ts(ts) if not isinstance(ts, datetime) else (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))
    return d.astimezone(LOCAL_TZ) if d is not None else None


def local_iso(ts) -> str:
    """'2026-11-09T15:30:00+05:00' for CSV files ('' if there is no time)."""
    d = to_local(ts)
    return d.isoformat(timespec="seconds") if d else ""


def local_day_start_utc(day: str) -> str:
    """Start of a Pakistan calendar day ('2026-11-09') as stored UTC text, for date filters."""
    d = datetime.combine(_date.fromisoformat(day), _time(0), tzinfo=LOCAL_TZ)
    return d.astimezone(timezone.utc).isoformat(timespec="seconds")


def local_day_end_utc(day: str) -> str:
    """Start of the NEXT Pakistan day as UTC text (use with '<')."""
    return local_day_start_utc((_date.fromisoformat(day) + timedelta(days=1)).isoformat())


def fmt_short(ts: Optional[str]) -> str:
    """'3:42 PM · 09 Nov' (Pakistan time) for tables."""
    d = to_local(ts)
    if d is None:
        return ""
    return f"{d.strftime('%I:%M %p').lstrip('0')} · {d.day:02d} {d.strftime('%b')}"


def fmt_time(ts: Optional[str], date: bool = True) -> str:
    """'3:30 PM, 09 Nov 2026' (Pakistan time). Empty string if there is no time."""
    d = to_local(ts)
    if d is None:
        return ""
    t = d.strftime("%I:%M %p").lstrip("0")
    return f"{t}, {d.day:02d} {d.strftime('%b %Y')}" if date else t


def machine_type_of(group: str) -> str:
    return group.split("_", 1)[0].title()


def machine_code(group: str) -> str:
    """'pump_id_04' -> 'PUMP-04' (how machine IDs are shown in the interface)."""
    parts = (group or "").split("_")
    return f"{parts[0].upper()}-{parts[-1]}" if len(parts) >= 2 else (group or "").upper()


def _pill(label: str, color: str, bg: str, large: bool = False) -> str:
    return (f'<span class="sl-pill{" lg" if large else ""}" style="color:{color};background:{bg}">'
            f'<i style="background:{color}"></i>{esc(label)}</span>')


def pill(state: str, large: bool = False) -> str:
    st = state if state in SEVERITY else "NO_DATA"
    return _pill(SEVERITY_LABEL.get(state, state), SEVERITY[st], SEVERITY_BG[st], large)


def clip_pill(level: str, large: bool = False) -> str:
    label, color = CLIP.get(level, ("UNKNOWN", SEVERITY["NO_DATA"]))
    bg = {"#3fb27f": SEVERITY_BG["NORMAL"], "#f0a31c": SEVERITY_BG["WARNING"], "#e5534b": SEVERITY_BG["HIGH"]}.get(color, SEVERITY_BG["NO_DATA"])
    return _pill(label, color, bg, large)


def status_pill(status: str, acknowledged: bool = False) -> str:
    if status == "open":
        return _pill("OPEN", SEVERITY["HIGH"], SEVERITY_BG["HIGH"])
    return _pill("CLOSED", "#a9b6c4", SEVERITY_BG["NO_DATA"])


def alert_pill(status: str) -> str:
    label, color, bg = ALERT_STATUS.get(status, (status.upper(), "#a9b6c4", "#26313e"))
    return _pill(label, color, bg)


def alert_text(a: dict) -> str:
    """One plain sentence about where an alert e-mail stands (used by the assistant and the reports)."""
    st = a["status"]
    if st == "sent":
        return f"sent to {a['recipient']} at {fmt_time(a['sent_utc'])} PKT"
    if st == "dry_run":
        return f"saved in test mode for {a['recipient']} at {fmt_time(a['sent_utc'])} PKT, NOT sent (no mail server configured)"
    if st == "pending":
        return f"waiting to be sent to {a['recipient']}" + (f" (last attempt failed: {a['last_error']})" if a.get("last_error") else "")
    if st == "failed":
        return f"FAILED to reach {a['recipient']} after {a['attempts']} attempts: {a['last_error']}"
    if st == "suppressed":
        return f"not sent to {a['recipient']} because e-mail alerts were switched off"
    return f"{st} ({a['recipient']})"


def plain_pill(label: str, color: str = "#a9b6c4", bg: str = "#26313e") -> str:
    return _pill(label, color, bg)


def source_tag(source: Optional[str]) -> str:
    return {"simulated": '<span class="sl-tag sim">SIMULATED</span>', "upload": '<span class="sl-tag up">UPLOAD</span>'}.get(source or "", "")


def banner(kind: str, html_text: str) -> str:
    """kind: high | warn | sim | info | ok. `html_text` must already be safe (use esc() on database values)."""
    return f'<div class="sl-banner {kind}">{html_text}</div>'


def kpi(label: str, value, detail: str = "", color: str = "var(--text)") -> str:
    return (f'<div class="sl-kpi"><div class="k">{esc(label)}</div><div class="v" style="color:{color}">{esc(value)}</div>'
            f'<div class="d">{esc(detail)}</div></div>')


def appbar(right_html: str = "") -> str:
    return (f'<div class="sl-appbar"><div><div class="name">{esc(PRODUCT)}</div><div class="sub">{esc(PRODUCT_SUB)}</div></div>'
            f'<div class="right">{right_html}</div></div>')


def stat_strip(items: list) -> str:
    """items: (label, value_html, colour or None, small=False)"""
    out = []
    for it in items:
        label, value, color = it[0], it[1], it[2]
        small = len(it) > 3 and it[3]
        dot = f'<i style="background:{color}"></i>' if color else ""
        out.append(f'<div class="sl-stat"><div class="k">{dot}{esc(label)}</div><div class="v{" sm" if small else ""}">{value}</div></div>')
    return f'<div class="sl-strip">{"".join(out)}</div>'


def kv_grid(items: list) -> str:
    """items: (label, value_html). Values are trusted HTML: escape database text with esc() before passing it in."""
    return '<div class="sl-kv">' + "".join(f'<div><div class="k">{esc(k)}</div><div class="v">{v}</div></div>' for k, v in items) + "</div>"


def facts(rows: list, left: bool = False) -> str:
    return f'<table class="sl-facts{" left" if left else ""}">' + "".join(f"<tr><td>{esc(k)}</td><td>{v}</td></tr>" for k, v in rows) + "</table>"


def table(headers: list, rows: list, row_classes: Optional[list] = None, num_cols: tuple = (), empty: str = "Nothing to show.") -> str:
    """headers: plain text. rows: lists of trusted HTML cells. num_cols: indexes of right-aligned numeric columns."""
    if not rows:
        return banner("info", esc(empty))
    th = "".join(f'<th{" class=num" if i in num_cols else ""}>{esc(h)}</th>' for i, h in enumerate(headers))
    body = []
    for i, r in enumerate(rows):
        cls = f' class="{row_classes[i]}"' if row_classes and row_classes[i] else ""
        body.append(f"<tr{cls}>" + "".join(f'<td{" class=num" if j in num_cols else ""}>{c}</td>' for j, c in enumerate(r)) + "</tr>")
    return f'<table class="sl-table"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def z_text(z) -> str:
    return "-" if z is None else f"{z:.1f}"


def mono(text) -> str:
    return f'<span class="mono">{esc(text)}</span>'
