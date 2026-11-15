"""Smart Listener - Industrial Acoustic Anomaly Detection: dashboard (Streamlit).

Start the API first, then, from the project folder:      streamlit run dashboard/app.py

Screens: Login | Overview | Analyze Audio | Machines | Incidents | Reports | Assistant | Settings
Everything goes through the backend API; nothing here opens the database. Look and feel: smart_listener.dashboard_ui
and .streamlit/config.toml. Every number shown comes from the backend; nothing is a placeholder.
"""
import json
import os
from datetime import date

import streamlit as st

from smart_listener import dashboard_ui as ui
from smart_listener.dashboard_client import ApiClient, ApiError

API_URL = os.environ.get("SL_API_URL", "http://127.0.0.1:8000")
WORKSPACE = [("overview", "Overview"), ("analyze", "Analyze Audio"), ("machines", "Machines"), ("incidents", "Incidents"),
             ("reports", "Reports"), ("assistant", "Assistant")]
ADMIN = [("settings", "Settings")]
SELECT_WHY = ("Each registered machine has its own trained model and alert limits, so a recording must be analysed "
              "with the model of the machine it came from. The machine ID also sets the machine type.")
SEC_COLOR = {"model": "#79aedb", "record": "#a9b6c4", "not_determined": "#f0a31c", "guidance": "#97a6b7", "action": "#3fb27f"}
SEC_SOURCE = {"model": "From the model", "record": "From the records", "not_determined": "", "guidance":
              "General practice, not from this incident's data", "action": "General practice"}

st.set_page_config(page_title=ui.PRODUCT_FULL, layout="wide", initial_sidebar_state="expanded")
st.markdown(ui.CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- plumbing
def client() -> ApiClient:
    return st.session_state["client"]


def call(fn, *args, **kw):
    """Run an API call; show a readable error instead of a crash. Returns None on failure."""
    try:
        return fn(*args, **kw)
    except ApiError as e:
        if e.status == 401 and st.session_state.get("client") is not None and st.session_state["client"].token:
            st.session_state.pop("client", None)
            st.warning("Your session expired. Please sign in again.")
            st.rerun()
        st.error(str(e))
        return None


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def goto(page: str, machine_id=None, incident_id=None, **extra):
    """Button callback: open a screen, optionally with a machine or incident selected."""
    st.session_state["page"] = page
    if machine_id:
        st.session_state["sel_machine"] = machine_id
    if incident_id:
        st.session_state["sel_incident"] = incident_id
    st.session_state.update(extra)


def page_head(title: str, sub: str, action=None):
    """Title, one-line description and an optional primary action (label, callback, args) on the right."""
    if action:
        c1, c2 = st.columns([5, 1.2], vertical_alignment="center")
        with c1:
            html(f'<div class="sl-title">{ui.esc(title)}</div><div class="sl-sub">{ui.esc(sub)}</div>')
        with c2:
            st.button(action[0], type="primary", key=f"head_{action[0]}", on_click=action[1], args=action[2],
                      use_container_width=True)
    else:
        html(f'<div class="sl-title">{ui.esc(title)}</div><div class="sl-sub">{ui.esc(sub)}</div>')


def card(key: str):
    return st.container(border=True, key=f"card-{key}")


def h(title: str, sub: str = ""):
    html(f'<div class="sl-h">{ui.esc(title)}</div>' + (f'<div class="sl-hsub">{ui.esc(sub)}</div>' if sub else
                                                        '<div style="height:10px"></div>'))


def th(cols, labels):
    for c, l in zip(cols, labels):
        c.markdown(f'<div class="sl-th">{ui.esc(l)}</div>', unsafe_allow_html=True)


def cell(col, content: str, strong=False):
    col.markdown(f'<div class="sl-cell{" strong" if strong else ""}">{content}</div>', unsafe_allow_html=True)


def code(group: str) -> str:
    return ui.machine_code(group)


def cached(key, fn, *args):
    """Values that do not change during a session (model profiles, system information)."""
    if key not in st.session_state:
        st.session_state[key] = call(fn, *args)
    return st.session_state[key] or {}


def profiles() -> dict:
    if "profiles" not in st.session_state:
        st.session_state["profiles"] = {p["group"]: p for p in call(client().profiles) or []}
    return st.session_state["profiles"]


def system() -> dict:
    return cached("system", client().system)


def limits():
    t = system().get("thresholds")
    return (t["z_warn"], t["z_alarm"]) if t else None


def machine_label(m) -> str:
    if m is None:
        return "Choose a registered machine"
    return f"{code(m['model_group'])}  ·  {m['name']}" + (f"  ·  {m['location']}" if m.get("location") else "")


def model_name(group: str) -> str:
    kind = profiles().get(group, {}).get("model")
    return f"Autoencoder ({kind})" if kind else "Autoencoder"


def z_colour(z):
    lim = limits()
    if z is None or not lim:
        return "var(--text)"
    return ui.SEVERITY["HIGH"] if z > lim[1] else ui.SEVERITY["WARNING"] if z > lim[0] else ui.SEVERITY["NORMAL"]


def running_simulations() -> list:
    return [s for s in call(client().sessions, None, 20) or [] if s["status"] == "running" and s["source"] == "simulated"]


def alert_banner():
    """Shown on every screen while a HIGH incident is open and not acknowledged."""
    inc = [i for i in call(client().incidents, "open", None, 20) or [] if not i["acknowledged"]]
    if not inc or st.session_state.get("page") == "incidents":
        return
    i = inc[0]
    more = f" (+{len(inc) - 1} more)" if len(inc) > 1 else ""
    c1, c2 = st.columns([6, 1], vertical_alignment="center")
    with c1:
        html(ui.banner("high", f"{ui.pill('HIGH')} &nbsp; <b>{ui.esc(code(i['model_group']))}</b> &nbsp;·&nbsp; "
                               f"{ui.inc_code(i['id'])} open since {ui.esc(ui.fmt_time(i['opened_utc']))} PKT "
                               f"&nbsp;·&nbsp; not acknowledged{ui.esc(more)}"))
    with c2:
        st.button("Review incident", key="banner_review", on_click=goto, args=("incidents", i["machine_id"], i["id"]),
                  use_container_width=True)


# ----------------------------------------------------------------------------- 1. login
def login_page():
    st.markdown("<style>[data-testid='stSidebar']{display:none}.block-container{padding-top:6vh !important}</style>",
                unsafe_allow_html=True)
    left, gap, right = st.columns([1.05, 0.15, 1], vertical_alignment="center")
    with left:
        html(f'<div style="padding:8vh 0 0"><div style="font-size:34px;font-weight:500;color:var(--text)">{ui.esc(ui.PRODUCT)}</div>'
             f'<div style="font-size:13px;color:var(--link);letter-spacing:1.3px;text-transform:uppercase;margin-top:8px">'
             f'{ui.esc(ui.PRODUCT_SUB)}</div><div style="font-size:18px;color:#c3cfdb;line-height:1.6;margin-top:48px;max-width:520px">'
             f'{ui.esc(ui.DESCRIPTION)}</div></div>')
    with right:
        with card("login"):
            html('<div style="font-size:28px;font-weight:500;color:var(--text);margin-bottom:6px">Sign in</div>'
                 '<div class="sl-hsub">Sign in with your administrator account.</div>')
            with st.form("login", border=False):
                user = st.text_input("Username", placeholder="Your username")
                pw = st.text_input("Password", type="password", placeholder="Enter your password")
                if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                    c = st.session_state.get("_client") or ApiClient(API_URL)
                    try:
                        c.login(user, pw)
                        st.session_state["client"] = c
                        st.rerun()
                    except ApiError as e:
                        st.error(str(e))
            html('<div class="sl-help" style="margin-top:14px">Access is restricted to authorised users. '
                 'A forgotten password is reset by the system administrator with scripts/init_db.py.</div>')


# ----------------------------------------------------------------------------- 2. fleet overview
def overview_page():
    page_head("Fleet overview", "See which machines are monitored and whether anything needs attention.",
              ("Analyze audio", goto, ("analyze",)))
    c = client()
    fleet = call(c.fleet) or []
    if not fleet:
        with card("empty-fleet"):
            h("No machines registered", "Register the machines you want to monitor under Machines.")
            st.button("Add machine", type="primary", on_click=goto, args=("machines",), kwargs={"add_machine": True})
        return
    open_inc = call(c.incidents, "open", None, 100) or []
    n = {s: sum(1 for m in fleet if m["current_state"] == s) for s in ui.SEVERITY}
    fans = sum(1 for m in fleet if m["machine_type"] == "fan")
    k = st.columns(4)
    k[0].markdown(ui.kpi("Registered machines", len(fleet), f"{fans} fans, {len(fleet) - fans} pumps"
                         + (f" · {n['NO_DATA']} not analysed" if n["NO_DATA"] else ""), "#79aedb"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Normal", n["NORMAL"], "Within normal range", ui.SEVERITY["NORMAL"]), unsafe_allow_html=True)
    att = n["WATCH"] + n["WARNING"] + n["HIGH"]
    k[2].markdown(ui.kpi("Needs attention", att, "Watch, warning or high",
                         ui.SEVERITY["WARNING"] if att else "var(--text)"), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Open incidents", len(open_inc), "Operator review needed" if open_inc else "None open",
                         ui.SEVERITY["HIGH"] if open_inc else "var(--text)"), unsafe_allow_html=True)
    html('<div style="height:22px"></div>')
    left, right = st.columns([2.3, 1], gap="medium")
    with left, card("machine-status"):
        h("Machine status", "Select a machine to view its latest analysis.")
        w = [1.0, 0.65, 1.05, 1.55, 0.7, 1.75, 1.0]
        th(st.columns(w), ["Machine ID", "Type", "Severity", "Meaning", "Level z", "Last analysed (PKT)", ""])
        for m in sorted(fleet, key=lambda x: (ui.SEVERITY_ORDER.get(x["current_state"], 9), x["model_group"])):
            with st.container(key=f"row-fleet-{m['id']}"):
                cols = st.columns(w, vertical_alignment="center")
                cell(cols[0], ui.esc(code(m["model_group"])), True)
                cell(cols[1], ui.esc(m["machine_type"].title()))
                cell(cols[2], ui.pill(m["current_state"]))
                cell(cols[3], f'<span style="color:var(--muted)">{ui.STATE_TEXT[m["current_state"]]}</span>')
                cell(cols[4], f'<span style="color:{z_colour(m["last_z"])}">{ui.z_text(m["last_z"])}</span>')
                cell(cols[5], (ui.esc(ui.fmt_short(m["last_window_utc"])) + " " + ui.source_tag(m["last_source"]))
                     if m["last_window_utc"] else '<span class="sl-dim">-</span>')
                cols[6].button("Details →", key=f"fv_{m['id']}", type="tertiary", on_click=goto, args=("machines", m["id"]))
        lim = limits()
        if lim:
            html(f'<div class="sl-help" style="margin-top:12px">Level z: how far the latest 10-second window is from the '
                 f'machine\'s normal sound. Warning above {lim[0]:.2f}, alarm above {lim[1]:.2f}.</div>')
    with right:
        with card("attention"):
            h("Needs attention")
            watch = sorted([m for m in fleet if m["current_state"] in ("HIGH", "WARNING", "WATCH") and not m["open_incident_id"]],
                           key=lambda x: ui.SEVERITY_ORDER[x["current_state"]])
            if not open_inc and not watch:
                html('<div class="sl-help">Nothing needs attention. All analysed machines are within their normal range.</div>')
            for i in open_inc[:3]:
                html(f'<div style="margin:2px 0 8px">{ui.pill("HIGH")} &nbsp;<span class="sl-cell strong">'
                     f'{ui.esc(code(i["model_group"]))}</span></div><div class="sl-help">{ui.inc_code(i["id"])} open since '
                     f'{ui.esc(ui.fmt_time(i["opened_utc"]))} PKT. Sustained deviation from this machine\'s normal sound'
                     f'{"" if i["acknowledged"] else "; not acknowledged"}.</div>')
                st.button("Review incident", key=f"att_{i['id']}", on_click=goto, args=("incidents", i["machine_id"], i["id"]))
            for m in watch[:4]:
                html(f'<div style="margin:10px 0 4px">{ui.pill(m["current_state"])} &nbsp;<span class="sl-cell strong">'
                     f'{ui.esc(code(m["model_group"]))}</span> <span class="sl-dim">· {ui.STATE_TEXT[m["current_state"]].lower()}'
                     f'{", no open incident" if m["current_state"] == "HIGH" else ""}</span></div>')
                st.button("Open machine", key=f"attm_{m['id']}", type="tertiary", on_click=goto, args=("machines", m["id"]))
        html('<div style="height:14px"></div>')
        with card("activity"):
            h("Recent activity")
            act = call(c.activity, 8) or []
            if not act:
                html('<div class="sl-help">No severity changes yet.</div>')
            rows = []
            for a in act:
                what = ("HIGH incident opened" if a["kind"] == "incident_opened" else
                        f"Returned to {a['to_state']}" if a["to_state"] == "NORMAL" else f"Moved to {a['to_state']}")
                rows.append(f'<div class="sl-act"><div class="t">{ui.esc(ui.fmt_time(a["time_utc"], False))}</div><div>'
                            f'{ui.esc(code(a["model_group"]))} · {ui.esc(what)}</div></div>')
            html("".join(rows))
            if any(a["source"] == "simulated" for a in act):
                html('<div class="sl-help" style="margin-top:8px"><span class="sl-tag sim">SIMULATED</span> &nbsp;These '
                     'events come from replayed MIMII recordings.</div>')
    st.button("Refresh", key="refresh_overview")


# ----------------------------------------------------------------------------- 3. analyze audio
def evidence_facts(ev: dict) -> str:
    summ = (ev or {}).get("summary")
    if not summ:
        return ""
    rows = []
    for b in summ["top_bands"]:
        w = max(2, int(b["share_of_error"] * 100 * 2))
        rows.append((f"{b['hz_low']} to {b['hz_high']} Hz",
                     f'{b["share_of_error"] * 100:.0f}%&nbsp;&nbsp;<span class="sl-bar" style="display:inline-block;width:70px;'
                     f'vertical-align:middle"><span style="width:{min(w, 100)}%"></span></span>'))
    t = f"{summ['peak_time_s']} s" if summ.get("peak_time_error_ratio", 1) >= 2 else "spread through the clip"
    return ('<div class="sl-label" style="margin-top:16px">Where the sound differs most</div>' + ui.facts(rows) +
            f'<div class="sl-help" style="margin-top:8px">Strongest in time: {ui.esc(t)}.</div>')


def identification_panel(file, machines, status):
    """Returns the machine id to preselect (or None) and shows how it was found."""
    key = f"{file.name}|{file.size}"
    by_group = {m["model_group"]: m for m in machines}
    if status.get("enabled"):
        store = st.session_state.setdefault("ident", {})
        if key not in store:
            with st.spinner("Comparing the recording with each machine's model"):
                store[key] = call(client().identify, file.name, file.getvalue())
        r = store.get(key)
        if not r:
            return None
        sug = r["suggestion"]
        if sug["group"]:
            html(ui.banner("ok", f"<b>Identified by sound: {ui.esc(code(sug['group']))}.</b> {ui.esc(sug['reason'])} "
                                 f"Check the machine below before running the analysis."))
        else:
            html(ui.banner("info", f"<b>Machine not identified.</b> {ui.esc(sug['reason'])}"))
        with st.expander("How well each machine's model fits this recording"):
            html(ui.table(["Machine ID", "Median level z", ""], [[ui.esc(code(x["group"])), f"{x['z']:.1f}",
                          "registered" if x["group"] in by_group else '<span class="dim">not registered</span>']
                          for x in r["ranking"]], num_cols=(1,)))
            html('<div class="sl-help">Lower means more like that machine\'s normal sound. An abnormal recording can be '
                 'unlike every machine, including its own, which is why low-confidence matches are not suggested.</div>')
        return sug.get("machine_id")
    from smart_listener.inference import filename_hint
    hint = filename_hint(file.name)
    text = f"<b>Select the machine.</b> {ui.esc(status.get('reason', ''))}"
    pre = None
    if hint:
        g = f"{hint['machine_type']}_{hint['machine_id']}"
        if g in by_group:
            pre = by_group[g]["id"]
            text += f" The file name suggests <b>{ui.esc(code(g))}</b>; this comes from the name, not from the sound."
    html(ui.banner("info", text))
    return pre


def analyze_page():
    page_head("Analyze audio", "Upload a WAV recording and compare it with the machine's learned normal sound.")
    c = client()
    machines = call(c.machines, False) or []
    left, right = st.columns([1.9, 1], gap="medium")
    with right, card("next"):
        h("What happens next")
        html('<div class="sl-step"><div class="n">01</div><div><div class="t">Audio is checked</div><div class="d">File type, '
             'duration, sample rate and silence.</div></div></div>'
             '<div class="sl-step"><div class="n">02</div><div><div class="t">The machine is chosen</div><div class="d">'
             'Suggested only where identification is validated; otherwise you select it.</div></div></div>'
             '<div class="sl-step"><div class="n">03</div><div><div class="t">Sound is analysed</div><div class="d">'
             'The machine\'s own model scores every 10-second window.</div></div></div>'
             '<div class="sl-step"><div class="n">04</div><div><div class="t">Evidence is shown</div><div class="d">'
             'Score, limits, severity and where the sound differs.</div></div></div>')
    with left, card("upload"):
        h("Choose an audio file", "A WAV recording of a registered fan or pump, at least 10 seconds long. The audio is not kept.")
        if not machines:
            html(ui.banner("info", "No machines are registered yet. Register one under Machines first."))
            return
        f = st.file_uploader("Drag and drop a .wav file here", type=["wav"], key="an_file")
        status = cached("ident_status", c.identification_status) if f is not None else {}
        pre = identification_panel(f, machines, status) if f is not None else None
        pre = pre or st.session_state.get("analyze_machine")
        ids = [m["id"] for m in machines]
        opts = [None] + machines
        a, b = st.columns([1.3, 1])
        with a:
            m = st.selectbox("Machine", opts, index=(ids.index(pre) + 1) if pre in ids else 0, format_func=machine_label,
                             key=f"an_machine_{pre}")
        with b:
            st.text_input("Input format", f"WAV · {cfg_rate()} mono preferred", disabled=True)
        html(f'<div class="sl-help">{ui.esc(SELECT_WHY)}</div>')
        go = st.button("Run analysis", type="primary", disabled=f is None or m is None, key="an_go")
        if go:
            with st.spinner("Analysing the recording"):
                r = call(c.upload, m["id"], f.name, f.getvalue())
            if r:
                st.session_state["an_result"] = r
                st.session_state["analyze_machine"] = m["id"]
    r = st.session_state.get("an_result")
    if r:
        html('<div style="height:18px"></div>')
        result_card(r)


def cfg_rate() -> str:
    sr = system().get("sample_rate")
    return f"{sr / 1000:g} kHz" if sr else "original rate"


def email_note(e):
    """One line under the result: was the owner e-mailed about this upload?"""
    if not e:
        return
    st_ = e.get("status")
    if e.get("needed") is False:
        html(f'<div class="sl-help" style="margin-bottom:10px">E-mail: not needed ({ui.esc(e["reason"])}).</div>')
    elif st_ == "pending":
        html(ui.banner("info", f'<b>E-mail to {ui.esc(e["recipient"])}</b> with the PDF report, the CSV of all windows and '
                               f'the evidence picture is being sent ({ui.esc(e["reason"])}). Its status is under '
                               f'Settings &rarr; E-mail delivery.'))
    elif st_ == "suppressed":
        html(ui.banner("info", "E-mail alerts are switched off in Settings, so this result was recorded but not e-mailed."))
    elif st_ == "duplicate":
        html(ui.banner("info", "This analysis has already been e-mailed."))
    elif st_ == "error":
        html(ui.banner("warn", "<b>E-mail not prepared.</b> " + ui.esc(e.get("reason", ""))))


def result_card(r):
    s, w = r["summary"], r["windows"]
    worst = w[r["worst_window"]]
    grp = r["machine"]["model_group"]
    many = not s["file_state_note"]
    with card("result"):
        top, tag = st.columns([4, 1], vertical_alignment="center")
        with top:
            badge = ui.pill(s["file_state"], True) if many else ui.clip_pill(worst["clip_level"], True)
            html(f'<div class="sl-h" style="margin-bottom:10px">Analysis result</div>{badge} &nbsp; '
                 f'<span style="font-size:19px;color:var(--text)">{ui.esc(code(grp))} · {ui.esc(ui.machine_type_of(grp))}</span>'
                 f' &nbsp;<span class="sl-dim">{ui.esc(r["file"]["name"])}</span>')
        with tag:
            st.button("Clear result", key="an_clear", on_click=lambda: st.session_state.pop("an_result", None))
        text = {"above_alarm_limit": "The recording differs strongly from this machine's learned normal sound.",
                "above_warning_limit": "The recording differs somewhat from this machine's learned normal sound.",
                "within_normal": "The recording is within this machine's normal range."}[worst["clip_level"]]
        html(f'<div class="sl-help" style="font-size:14.5px;color:#c3cfdb;margin:14px 0 16px">{text}</div>')
        for note in r["warnings"]:
            html(ui.banner("warn", ui.esc(note)))
        if r.get("model_caution"):
            html(ui.banner("warn", "<b>Model reliability.</b> " + ui.esc(r["model_caution"])))
        email_note(r.get("email"))
        sev = (ui.pill(s["file_state"]) + f'<div class="sl-help">Judged over {s["n_windows"]} consecutive windows.</div>') if many else \
            ('<span class="sl-dim">Not assessed</span><div class="sl-help">Severity needs several consecutive windows; '
             f'this file has {s["n_windows"]}. The window result is shown instead.</div>')
        k1, k2, k3 = st.columns(3, gap="medium")
        with k1, card("score"):
            html(f'<div class="sl-label">Anomaly score</div><div style="font-size:34px;color:{z_colour(worst["z"])};margin:4px 0">'
                 f'z = {worst["z"]:.1f}</div>' + ui.facts([("Reconstruction error (MSE)", f"{worst['score']:.4f}"),
                                                          ("Warning limit", f"z > {worst['z_warn']:.2f}"),
                                                          ("Alarm limit", f"z > {worst['z_alarm']:.2f}"),
                                                          ("Windows above alarm", f"{s['n_above_alarm_limit']} of {s['n_windows']}")]))
        with k2, card("severity"):
            html('<div class="sl-label">Operational severity</div><div style="margin:8px 0 6px">' + sev + '</div>'
                 '<div class="sl-label" style="margin-top:14px">Highest window</div><div class="sl-cell">'
                 f'{worst["offset_s"]:.0f} s to {worst["offset_s"] + 10:.0f} s &nbsp;{ui.clip_pill(worst["clip_level"])}</div>')
        with k3, card("limits"):
            html('<div class="sl-label">What the model can tell you</div><div class="sl-help" style="font-size:14px;color:#c3cfdb">'
                 'It detects how far the sound is from this machine\'s normal sound. It cannot identify the physical '
                 'fault, predict failure time or give a probability of failure.</div>')
        t1, t2 = st.tabs(["Audio evidence", "All windows"])
        with t1:
            e1, e2 = st.columns([3, 1.2], gap="medium")
            with e1:
                try:
                    st.image(client().evidence_png(worst["analysis_id"]))
                except ApiError as e:
                    st.warning(str(e))
            with e2:
                html(evidence_facts(worst.get("evidence")) or '<div class="sl-help">No evidence summary.</div>')
                html('<div class="sl-help" style="margin-top:10px">Waveform: loudness over time. Spectrogram: frequencies '
                     'over time. Residual heatmap: where the model could not reproduce the sound; bright means most '
                     'different from normal. It does not name a cause.</div>')
        with t2:
            html(ui.table(["#", "Start (s)", "Level z", "Score (MSE)", "Result"],
                          [[str(i + 1), f"{x['offset_s']:.0f}", f"<b>{x['z']:.1f}</b>", f"{x['score']:.4f}", ui.clip_pill(x["clip_level"])]
                           for i, x in enumerate(w)],
                          ["high" if x["clip_level"] == "above_alarm_limit" else "warning" if x["clip_level"] == "above_warning_limit" else ""
                           for x in w], num_cols=(1, 2, 3)))
        html('<div class="sl-help" style="margin-top:8px">An uploaded recording never changes a machine\'s status and never '
             'opens an incident.</div>')


# ----------------------------------------------------------------------------- 4. machines
def register_form(machines):
    prof = profiles()
    with card("add-machine"):
        h("Add machine", "The machine ID selects the trained model and alert limits; the type follows from it.")
        n = st.session_state.get("form_n", 0)      # a new number gives a fresh, empty form after a successful save
        taken = {m["model_group"] for m in machines if m["active"]}
        groups = list(prof) or sorted(system().get("models", {}))
        free = [g for g in groups if g not in taken]
        group = st.selectbox("Machine ID", groups, index=groups.index(free[0]) if free else 0, format_func=code,
                             key=f"nm_group_{n}")
        reg = [m["name"] for m in machines if m["model_group"] == group]
        html(f'<div class="sl-help">Machine type: <b>{ui.esc(ui.machine_type_of(group))}</b>. '
             + (f'Already registered as {ui.esc(", ".join(reg))}.' if reg else "") + '</div>')
        with st.form(f"new_machine_{n}", border=False):
            a, b = st.columns(2)
            name = a.text_input("Name", key=f"nm_name_{n}")
            location = b.text_input("Location", key=f"nm_loc_{n}")
            owner = a.text_input("Owner name", key=f"nm_owner_{n}")
            email = b.text_input("Owner e-mail (alerts go here)", key=f"nm_email_{n}")
            notes = st.text_area("Notes", "", key=f"nm_notes_{n}", height=80)
            s1, s2, _ = st.columns([1, 1, 4])
            save = s1.form_submit_button("Register machine", type="primary")
            cancel = s2.form_submit_button("Cancel")
        if cancel:
            st.session_state["add_machine"] = False
            st.rerun()
        if save:
            r = call(client().create_machine, name=name, machine_type=group.split("_")[0], model_group=group,
                     location=location, owner_name=owner, owner_email=email, notes=notes)
            if r:
                st.session_state.update(form_n=n + 1, add_machine=False, sel_machine=r["id"],
                                        flash=f"Registered {r['name']} ({code(r['model_group'])}).")
                st.rerun()      # on an error the form keeps what you typed


def machines_page():
    page_head("Machines", "Registered machines, their monitoring state and history.",
              ("Add machine", lambda: st.session_state.update(add_machine=True), ()))
    if st.session_state.get("flash"):
        html(ui.banner("ok", ui.esc(st.session_state.pop("flash"))))
    c = client()
    machines = call(c.machines, True) or []
    if st.session_state.get("add_machine") or not machines:
        register_form(machines)
        html('<div style="height:16px"></div>')
    if not machines:
        return
    fleet = {f["id"]: f for f in call(c.fleet, True) or []}
    with card("registered"):
        h("Registered machines")
        w = [1.05, 0.7, 1.25, 1.25, 1.0, 1.9, 0.7]
        th(st.columns(w), ["Machine ID", "Type", "Name", "Model", "State", "Last analysis (PKT)", ""])
        for m in sorted(machines, key=lambda x: (not x["active"], x["model_group"])):
            f = fleet.get(m["id"], {})
            with st.container(key=f"row-mach-{m['id']}"):
                cols = st.columns(w, vertical_alignment="center")
                cell(cols[0], ui.esc(code(m["model_group"])), True)
                cell(cols[1], ui.esc(m["machine_type"].title()))
                cell(cols[2], ui.esc(m["name"]) + ("" if m["active"] else ' <span class="sl-dim">(inactive)</span>'))
                cell(cols[3], ui.esc(model_name(m["model_group"])))
                cell(cols[4], ui.pill(f.get("current_state", "NO_DATA")))
                cell(cols[5], (ui.esc(ui.fmt_short(f["last_window_utc"])) + " " + ui.source_tag(f.get("last_source")))
                     if f.get("last_window_utc") else '<span class="sl-dim">Not analysed yet</span>')
                cols[6].button("Open →", key=f"mo_{m['id']}", type="tertiary",
                               on_click=lambda i=m["id"]: st.session_state.update(sel_machine=i))
    ids = [m["id"] for m in machines]
    cur = st.session_state.get("sel_machine")
    if cur not in ids:
        cur = sorted(machines, key=lambda x: (ui.SEVERITY_ORDER.get(fleet.get(x["id"], {}).get("current_state", "NO_DATA"), 9),
                                              x["model_group"]))[0]["id"]
    m = next(x for x in machines if x["id"] == cur)
    html('<div style="height:18px"></div>')
    machine_detail(m, fleet.get(m["id"], {}))


def machine_detail(m, f):
    c = client()
    state = f.get("current_state", "NO_DATA")
    grp = m["model_group"]
    with card("machine-detail"):
        t1, t2 = st.columns([4, 1], vertical_alignment="center")
        t1.markdown(f'<div class="sl-h">Machine detail · {ui.esc(code(grp))}</div>', unsafe_allow_html=True)
        t2.markdown(f'<div style="text-align:right">{ui.pill(state, True)}</div>', unsafe_allow_html=True)
        lim = limits()
        if f.get("last_window_utc"):
            text = (f"{ui.STATE_TEXT[state]}. The latest window scored z = {f['last_z']:.1f}"
                    + (f" (warning above {lim[0]:.2f}, alarm above {lim[1]:.2f})" if lim else "")
                    + f", compared with {ui.esc(code(grp))}'s learned normal sound.")
        else:
            text = "This machine has not been analysed yet. Upload a recording or run simulated monitoring."
        html(f'<div class="sl-help" style="font-size:14.5px;color:#c3cfdb;margin:8px 0 14px">{text}</div>')
        p = profiles().get(grp, {})
        auc = f"{p['test_auc']:.3f}" if p.get("test_auc") is not None else "-"
        html(ui.kv_grid([("Machine ID", ui.esc(code(grp))), ("Type", ui.esc(ui.machine_type_of(grp))), ("Name", ui.esc(m["name"])),
                         ("Location", ui.esc(m["location"]) or "-"), ("Owner", ui.esc(m["owner_name"])),
                         ("Alert e-mail", ui.esc(m["owner_email"])), ("Model", ui.esc(model_name(grp))),
                         ("Held-out test ROC-AUC", ui.esc(auc)),
                         ("Open incident", ui.inc_code(f.get("open_incident_id")) if f.get("open_incident_id") else "None"),
                         ("Incidents in total", str(f.get("n_incidents", 0)))]))
        if p.get("caution"):
            html(ui.banner("warn", "<b>Model reliability.</b> " + ui.esc(p["caution"])))
        b1, b2, _ = st.columns([1.2, 1.3, 3])
        b1.button("Analyze this machine", type="primary", key="md_an", disabled=not m["active"],
                  on_click=goto, args=("analyze",), kwargs={"analyze_machine": m["id"], "an_result": None})
        b2.button("View incident history", key="md_inc", on_click=goto, args=("incidents", m["id"]),
                  kwargs={"inc_machine": m["id"]})
        tabs = st.tabs(["History", "Simulated monitoring", "Analysed windows", "Edit machine"])
        with tabs[0]:
            history_tab(m)
        with tabs[1]:
            monitoring_tab(m)
        with tabs[2]:
            windows_tab(m)
        with tabs[3]:
            edit_tab(m)


def history_tab(m):
    c = client()
    sessions = [x for x in call(c.sessions, m["id"], 30) or [] if x["source"] == "simulated"]
    pick = st.selectbox("Show", [None] + sessions, key=f"hsess_{m['id']}",
                        format_func=lambda x: "All simulated monitoring windows" if x is None else
                        f"Session #{x['id']} · {ui.fmt_time(x['started_utc'])} · {x['status']}")
    try:
        st.image(c.history_png(m["id"], pick["id"] if pick else None))
    except ApiError as e:
        st.error(str(e))
    ev = call(c.severity_events, m["id"], 30) or []
    html('<div class="sl-label" style="margin-top:12px">Severity changes</div>' +
         ui.table(["Time (PKT)", "From", "To", "Reason"],
                  [[ui.esc(ui.fmt_time(e["event_utc"])), ui.pill(e["from_state"]), ui.pill(e["to_state"]), ui.esc(e["reason"])] for e in ev],
                  empty="No severity changes yet."))


@st.fragment(run_every=2)
def monitoring_progress(session_id: int, machine_id: int):
    c = client()
    s = call(c.session, session_id)
    if not s:
        return
    p = s.get("progress") or {}
    done, total = p.get("done", 0), max(p.get("total", 1), 1)
    latest = p.get("latest") or {}
    html(ui.stat_strip([("Session", f"#{session_id}", None), ("Status", s["status"].upper(), None),
                        ("Windows", f"{done} / {total}", None),
                        ("Current severity", ui.pill(latest["state"], True) if latest else "-", None, True),
                        ("Level z", f"{latest['z']:.1f}" if latest else "-", None)]))
    st.progress(min(done / total, 1.0))
    try:
        st.image(c.history_png(machine_id, session_id))
    except ApiError:
        pass
    if s["status"] != "running":
        st.session_state.pop("running_session", None)
        st.session_state["last_session"] = session_id
        if s["status"] == "error":
            st.error(s["note"])
        st.rerun()


def monitoring_tab(m):
    c = client()
    html(ui.banner("sim", '<span class="sl-tag sim">SIMULATED MONITORING</span> &nbsp;Replays recorded MIMII dataset clips one '
                          'after another as if they came from a microphone. It is not a live connection to a machine.'))
    if not m["active"]:
        html(ui.banner("info", "This machine is inactive. Activate it under Edit machine to monitor it."))
        return
    running = st.session_state.get("running_session")
    if running and running.get("machine_id") == m["id"]:
        monitoring_progress(running["id"], m["id"])
        if st.button("Stop monitoring", key=f"stop_{m['id']}"):
            call(c.stop_monitoring, running["id"])
            st.session_state.pop("running_session", None)
            st.rerun()
        return
    sc = cached("scenarios", c.scenarios) or {"scenarios": {}}
    keys = list(sc.get("scenarios", {}))
    if not keys:
        return
    scenario = st.selectbox("Scenario", keys, index=keys.index("fault") if "fault" in keys else 0, key=f"sc_{m['id']}",
                            format_func=lambda k: sc["scenarios"][k][0].upper() + sc["scenarios"][k][1:])
    nb, nf, na, nw = 12, 24, 12, 30
    if scenario == "normal":
        nw = st.number_input("Number of normal clips", 1, 200, 30, key=f"nw_{m['id']}")
    else:
        a, b, d = st.columns(3)
        nb = a.number_input("Normal clips first", 0, 200, 12, key=f"nb_{m['id']}")
        nf = b.number_input("Abnormal clips", 0, 200, 24, key=f"nf_{m['id']}")
        if scenario == "fault_recovery":
            na = d.number_input("Normal clips after", 0, 200, 12, key=f"na_{m['id']}")
    speed = (call(c.settings) or {}).get("replay_seconds_per_window", 1.0)
    html(f'<div class="sl-help">Each clip is 10 seconds of sound, replayed every {speed} real seconds (Settings). '
         f'Clips come from the validation or test data only.</div>')
    if st.button("Start simulated monitoring", type="primary", key=f"start_{m['id']}"):
        s = call(c.start_monitoring, m["id"], scenario=scenario, n_normal_before=int(nb), n_fault=int(nf),
                 n_normal_after=int(na), n_windows=int(nw))
        if s:
            st.session_state["running_session"] = {"id": s["id"], "machine_id": m["id"]}
            st.rerun()
    last = st.session_state.get("last_session")
    if last:
        s = call(c.session, last)
        if s and s["machine_id"] == m["id"]:
            html(ui.banner("ok" if s["status"] == "finished" else "warn",
                           f"Simulated session #{last} {ui.esc(s['status'])}. The History tab shows the result."))


def windows_tab(m):
    c = client()
    an = call(c.analyses, m["id"], 100) or []
    html(ui.table(["Time (PKT)", "Source", "Level z", "Score (MSE)", "Window result", "Severity"],
                  [[ui.esc(ui.fmt_time(a["window_utc"])), ui.source_tag(a["source"]), f"<b>{a['z']:.1f}</b>", f"{a['score']:.4f}",
                    ui.clip_pill(a["clip_level"]), ui.pill(a["state"]) if a["state"] else '<span class="dim">-</span>'] for a in an],
                  ["high" if a["clip_level"] == "above_alarm_limit" else "" for a in an], num_cols=(2, 3),
                  empty="No windows analysed yet."))
    flagged = [a for a in an if '"figure"' in (a["evidence_json"] or "")]
    if flagged:
        pick = st.selectbox("Evidence for window", flagged, key=f"ev_{m['id']}",
                            format_func=lambda a: f"{ui.fmt_time(a['window_utc'])} · z = {a['z']:.1f} · {a['source']}")
        e1, e2 = st.columns([3, 1.2], gap="medium")
        with e1:
            try:
                st.image(c.evidence_png(pick["id"]))
            except ApiError as e:
                st.warning(str(e))
        with e2:
            try:
                ev = json.loads(pick["evidence_json"] or "{}")
            except ValueError:
                ev = {}
            html(ui.facts([("Level z", f"{pick['z']:.1f}"), ("Score (MSE)", f"{pick['score']:.4f}"),
                           ("Warning / alarm limit", f"{pick['z_warn']:.2f} / {pick['z_alarm']:.2f}")]) + evidence_facts(ev))


def edit_tab(m):
    with st.form(f"edit_machine_{m['id']}", border=False):
        a, b = st.columns(2)
        n_name = a.text_input("Name", m["name"])
        n_loc = b.text_input("Location", m["location"])
        n_owner = a.text_input("Owner name", m["owner_name"])
        n_email = b.text_input("Owner e-mail", m["owner_email"])
        n_active = st.checkbox("Active (inactive machines are not monitored)", bool(m["active"]))
        if st.form_submit_button("Save changes", type="primary"):
            if call(client().update_machine, m["id"], name=n_name, owner_name=n_owner, owner_email=n_email,
                    location=n_loc, active=n_active):
                st.session_state["flash"] = "Changes saved."
                st.rerun()


# ----------------------------------------------------------------------------- 5. incidents
INC_STATUS = {"All": None, "Open": "open", "Closed": "closed", "Not acknowledged": "unack"}


def incidents_page():
    page_head("Incidents", "An incident opens when a machine's sound stays far from normal (HIGH severity).")
    c = client()
    machines = call(c.machines, True) or []
    f1, f2, _ = st.columns([1, 1.3, 1.7])
    status = f1.selectbox("Status", list(INC_STATUS), key="inc_status")
    mids = [None] + [m["id"] for m in machines]
    names = {m["id"]: code(m["model_group"]) for m in machines}
    pre = st.session_state.get("inc_machine")
    mid = f2.selectbox("Machine", mids, index=mids.index(pre) if pre in mids else 0, key=f"inc_machine_{pre}",
                       format_func=lambda i: "All machines" if i is None else names[i])
    st.session_state["inc_machine"] = mid
    s = INC_STATUS[status]
    inc = call(c.incidents, s if s in ("open", "closed") else None, mid, 500) or []
    if s == "unack":
        inc = [i for i in inc if not i["acknowledged"]]
    with card("incident-history"):
        h("Incident history")
        if not inc:
            html('<div class="sl-help">No incidents match these filters.</div>')
        else:
            w = [0.9, 1, 0.95, 1.3, 0.7, 1.0, 1.2, 1.0]
            th(st.columns(w), ["Incident", "Machine", "Severity", "Opened (PKT)", "Peak z", "Status", "Acknowledged", ""])
            for i in inc:
                with st.container(key=f"row-inc-{i['id']}"):
                    cols = st.columns(w, vertical_alignment="center")
                    cell(cols[0], ui.inc_code(i["id"]), True)
                    cell(cols[1], ui.esc(code(i["model_group"])))
                    cell(cols[2], ui.pill("HIGH"))
                    cell(cols[3], ui.esc(ui.fmt_short(i["opened_utc"])))
                    cell(cols[4], f"{i['peak_z']:.1f}")
                    cell(cols[5], ui.status_pill(i["status"]))
                    cell(cols[6], f"yes, {ui.esc(i['ack_by'])}" if i["acknowledged"] else f'<span style="color:{ui.SEVERITY["WARNING"]}">no</span>')
                    cols[7].button("Details →", key=f"iv_{i['id']}", type="tertiary",
                                   on_click=lambda x=i["id"]: st.session_state.update(sel_incident=x))
    if not inc:
        return
    ids = [i["id"] for i in inc]
    cur = st.session_state.get("sel_incident")
    sel = next((i for i in inc if i["id"] == cur), inc[0])
    html('<div style="height:18px"></div>')
    incident_detail(sel["id"])


def incident_detail(iid):
    c = client()
    full = call(c.incident, iid)
    if not full:
        return
    grp = full["model_group"]
    with card("incident-detail"):
        t1, t2 = st.columns([4, 1], vertical_alignment="center")
        t1.markdown(f'<div class="sl-h">{ui.inc_code(iid)} · {ui.esc(code(grp))}</div>', unsafe_allow_html=True)
        t2.markdown(f'<div style="text-align:right">{ui.pill("HIGH", True)}</div>', unsafe_allow_html=True)
        opening = call(c.analysis, full["opened_analysis_id"]) if full.get("opened_analysis_id") else None
        src = (opening or {}).get("source")
        text = (f"Sustained acoustic deviation from {ui.esc(code(grp))}'s learned normal sound"
                + (" during SIMULATED MONITORING (replayed MIMII recordings)" if src == "simulated" else "")
                + ". The model has not determined a physical fault type.")
        html(f'<div class="sl-help" style="font-size:14.5px;color:#c3cfdb;margin:8px 0 14px">{text}</div>')
        html(ui.kv_grid([("Machine", ui.esc(code(grp)) + f' <span class="sl-dim">· {ui.esc(full["machine_name"])}</span>'),
                         ("Machine type", ui.esc(ui.machine_type_of(grp))), ("Incident", ui.inc_code(iid)),
                         ("Highest severity", ui.pill("HIGH")), ("Start time (PKT)", ui.esc(ui.fmt_time(full["opened_utc"]))),
                         ("End time (PKT)", ui.esc(ui.fmt_time(full["closed_utc"])) or "ongoing"),
                         ("Current status", ui.status_pill(full["status"])),
                         ("Peak level z", f"{full['peak_z']:.1f}"), ("Windows at HIGH", str(full["n_windows"])),
                         ("Acknowledged", f"yes, by {ui.esc(full['ack_by'])} at {ui.esc(ui.fmt_time(full['ack_utc']))}"
                          if full["acknowledged"] else "no")]))
        if full["ack_note"]:
            html(ui.banner("ok", "<b>Acknowledgement note:</b> " + ui.esc(full["ack_note"])))
        act = st.session_state.get("inc_action")
        b = st.columns([1.35, 1.3, 1.2, 1.1, 1.4])
        if not full["acknowledged"]:
            b[0].button("Acknowledge incident", type="primary", key=f"a_ack_{iid}",
                        on_click=lambda: st.session_state.update(inc_action=("ack", iid)))
        b[1].button("Add operator note", key=f"a_note_{iid}", on_click=lambda: st.session_state.update(inc_action=("note", iid)))
        b[2].button("Download report", key=f"a_rep_{iid}", on_click=goto, args=("reports", None, iid),
                    kwargs={"rep_type": "Incident report (PDF)"})
        b[3].button("Ask assistant", key=f"a_ask_{iid}", on_click=goto, args=("assistant", None, iid))
        if act == ("ack", iid) and not full["acknowledged"]:
            with st.form(f"ack_form_{iid}", border=False):
                note = st.text_input("Note (optional)", max_chars=500)
                s1, s2, _ = st.columns([1, 1, 4])
                if s1.form_submit_button("Confirm acknowledgement", type="primary"):
                    if call(c.acknowledge, iid, note):
                        st.session_state.update(inc_action=None, inc_flash="Incident acknowledged.")
                        st.rerun()
                if s2.form_submit_button("Cancel"):
                    st.session_state["inc_action"] = None
                    st.rerun()
        if act == ("note", iid):
            with st.form(f"note_form_{iid}", border=False):
                note = st.text_area("Operator note", max_chars=1000, height=80)
                s1, s2, _ = st.columns([1, 1, 4])
                if s1.form_submit_button("Save note", type="primary"):
                    if call(c.add_note, iid, note):
                        st.session_state.update(inc_action=None, inc_flash="Note saved.")
                        st.rerun()
                if s2.form_submit_button("Cancel"):
                    st.session_state["inc_action"] = None
                    st.rerun()
        if st.session_state.get("inc_flash"):
            html(ui.banner("ok", ui.esc(st.session_state.pop("inc_flash"))))
        html('<div style="height:10px"></div>')
        l, r = st.columns([1.8, 1], gap="medium")
        with l:
            html('<div class="sl-label">Evidence for the window that opened the incident</div>')
            if full.get("opened_analysis_id"):
                try:
                    st.image(c.evidence_png(full["opened_analysis_id"]))
                except ApiError:
                    html('<div class="sl-help">No evidence picture was saved for this window.</div>')
            else:
                html('<div class="sl-help">No analysis is linked to this incident.</div>')
        with r:
            events = [e for e in call(c.severity_events, full["machine_id"], 200) or []
                      if e["event_utc"] >= full["opened_utc"][:19] and (not full["closed_utc"] or e["event_utc"] <= full["closed_utc"])]
            html('<div class="sl-label">Severity changes</div>' +
                 ui.table(["Time", "Change"], [[ui.esc(ui.fmt_time(e["event_utc"], False)), f"{ui.pill(e['from_state'])} → {ui.pill(e['to_state'])}"]
                                               for e in reversed(events)], empty="None recorded in this period."))
            al = full.get("alerts", [])
            html('<div class="sl-label" style="margin-top:16px">Notifications</div>' +
                 ui.table(["E-mail", "Status", "Time (PKT)"],
                          [[ui.esc(ui.ALERT_KIND.get(a["kind"], a["kind"])) + f'<div class="dim">{ui.esc(a["recipient"])}</div>',
                            ui.alert_pill(a["status"]), ui.esc(ui.fmt_short(a["sent_utc"] or a["created_utc"]))] for a in reversed(al)],
                          empty="No notification recorded."))
            if any(a["status"] == "dry_run" for a in al):
                html('<div class="sl-help">TEST MODE: no mail server is configured, so the message was saved as a file '
                     'and not sent. See Settings.</div>')
            if any(a["status"] == "failed" for a in al):
                html('<div class="sl-help">A FAILED e-mail can be put back in the queue under Settings.</div>')
            notes = call(c.notes, iid) or []
            html('<div class="sl-label" style="margin-top:16px">Operator notes</div>' +
                 (ui.table(["Time", "By", "Note"], [[ui.esc(ui.fmt_time(n["created_utc"])), ui.esc(n["author"]), ui.esc(n["note"])]
                                                    for n in notes]) if notes else '<div class="sl-help">No notes yet.</div>'))


# ----------------------------------------------------------------------------- 6. reports
REPORT_TYPES = ["Incident report (PDF)", "Machine report (PDF)", "Analysis data (CSV)", "Incident list (CSV)"]


def reports_page():
    page_head("Reports", "Export a record of an incident, a machine, or the analysed data.")
    c = client()
    machines = call(c.machines, True) or []
    incidents = call(c.incidents, None, None, 500) or []
    left, right = st.columns([1.5, 1], gap="medium")
    spec = {}
    with left, card("create-report"):
        h("Create a report")
        pre = st.session_state.get("rep_type")
        a, b = st.columns(2)
        rtype = a.selectbox("Report type", REPORT_TYPES, index=REPORT_TYPES.index(pre) if pre in REPORT_TYPES else 0,
                            key=f"rep_type_{pre}")
        st.session_state["rep_type"] = rtype
        mids = [m["id"] for m in machines]
        mlabel = {m["id"]: f"{code(m['model_group'])} · {m['name']}" for m in machines}
        if rtype == "Incident report (PDF)":
            if not incidents:
                html(ui.banner("info", "No incidents have been recorded, so there is nothing to report yet."))
                return
            iids = [i["id"] for i in incidents]
            cur = st.session_state.get("sel_incident")
            inc_l = {i["id"]: f"{ui.inc_code(i['id'])} · {code(i['model_group'])} · {i['status']}" for i in incidents}
            spec["incident"] = b.selectbox("Incident", iids, index=iids.index(cur) if cur in iids else 0,
                                           format_func=inc_l.get, key="rep_inc")
            spec["evidence"] = st.checkbox("Include the anomaly chart and evidence picture", True, key="rep_ev1")
        elif rtype == "Incident list (CSV)":
            spec["status"] = b.selectbox("Status", ["All", "open", "closed"], key="rep_st")
            spec["machine"] = st.selectbox("Machine", [None] + mids, key="rep_m2",
                                           format_func=lambda i: "All machines" if i is None else mlabel[i])
        else:
            if not machines:
                html(ui.banner("info", "No machines registered."))
                return
            opts = mids if rtype == "Machine report (PDF)" else [None] + mids
            cur = st.session_state.get("sel_machine")
            spec["machine"] = b.selectbox("Machine", opts, index=opts.index(cur) if cur in opts else 0, key=f"rep_m_{rtype}",
                                          format_func=lambda i: "All machines" if i is None else mlabel[i])
            d1, d2 = st.columns(2)
            spec["from"] = d1.date_input("From (PKT date, optional)", value=None, key="rep_from", format="YYYY-MM-DD")
            spec["to"] = d2.date_input("To (PKT date, optional)", value=None, key="rep_to", format="YYYY-MM-DD")
            if rtype == "Machine report (PDF)":
                spec["evidence"] = st.checkbox("Include the anomaly chart and evidence picture", True, key="rep_ev2")
        if st.button("Generate report", type="primary", key="rep_go"):
            with st.spinner("Building the report"):
                st.session_state["report_file"] = build_report(rtype, spec)
        rf = st.session_state.get("report_file")
        if rf:
            name, mime, data = rf
            html(ui.banner("ok", f"<b>{ui.esc(name)}</b> is ready ({len(data) / 1024:.0f} KB)."))
            st.download_button(f"Download {name}", data, file_name=name, mime=mime, type="primary", key="rep_dl")
    with right, card("preview"):
        h("Report contents")
        report_preview(rtype, spec, incidents, machines)


def _iso(d):
    return d.isoformat() if isinstance(d, date) else None


def build_report(rtype, spec):
    c = client()
    if rtype == "Incident report (PDF)":
        data = call(c.incident_pdf, spec["incident"], spec["evidence"])
        return (f"incident_{spec['incident']:03d}.pdf", "application/pdf", data) if data else None
    if rtype == "Machine report (PDF)":
        data = call(c.machine_pdf, spec["machine"], _iso(spec["from"]), _iso(spec["to"]), spec["evidence"])
        return (f"machine_{spec['machine']}.pdf", "application/pdf", data) if data else None
    if rtype == "Analysis data (CSV)":
        data = call(c.analyses_csv, spec["machine"], _iso(spec["from"]), _iso(spec["to"]))
        return ("analyses.csv", "text/csv", data) if data else None
    data = call(c.incidents_csv, None if spec["status"] == "All" else spec["status"], spec["machine"])
    return ("incidents.csv", "text/csv", data) if data else None


def report_preview(rtype, spec, incidents, machines):
    lines = [("Product", ui.esc(ui.PRODUCT_FULL))]
    if rtype == "Incident report (PDF)" and spec.get("incident"):
        i = next(x for x in incidents if x["id"] == spec["incident"])
        lines += [("Report", f"Incident summary · {ui.inc_code(i['id'])}"), ("Machine", ui.esc(code(i["model_group"]))),
                  ("Type", ui.esc(ui.machine_type_of(i["model_group"]))), ("Operational severity", "HIGH"),
                  ("Status", i["status"].upper()), ("Opened (PKT)", ui.esc(ui.fmt_time(i["opened_utc"]))),
                  ("Peak level z", f"{i['peak_z']:.1f}"), ("Acknowledged", "yes" if i["acknowledged"] else "no"),
                  ("Fault diagnosis", "Not determined")]
    elif rtype in ("Machine report (PDF)", "Analysis data (CSV)"):
        m = next((x for x in machines if x["id"] == spec.get("machine")), None)
        period = f"{_iso(spec.get('from')) or 'first record'} to {_iso(spec.get('to')) or 'latest record'}"
        lines += [("Report", "Machine summary" if rtype.startswith("Machine") else "One row per analysed 10-second window"),
                  ("Machine", ui.esc(code(m["model_group"])) if m else "All machines"), ("Period", ui.esc(period))]
        if rtype.startswith("Machine"):
            lines += [("Contains", "Windows analysed, highest level, limits, incidents, chart, evidence"),
                      ("Fault diagnosis", "Not determined")]
        else:
            lines += [("Columns", "time, source, score, z, limits, window result, severity")]
    else:
        lines += [("Report", "One row per incident"), ("Columns", "incident, machine, status, times, peak, acknowledgement")]
    html(ui.facts(lines, left=True) + '<div class="sl-help" style="margin-top:14px">The report records '
         'model measurements and system events. It does not claim a confirmed physical fault, a failure time or a '
         'failure probability. Simulated monitoring data is labelled as such.</div>')


# ----------------------------------------------------------------------------- 7. assistant
def assistant_page():
    page_head("Maintenance assistant", "Ask about an incident. Answers use the recorded evidence; general guidance is marked.")
    c = client()
    incidents = call(c.incidents, None, None, 200) or []
    if not incidents:
        with card("no-incidents"):
            h("No incidents yet", "The assistant answers questions about recorded incidents. None have been recorded.")
        return
    left, right = st.columns([1.8, 1], gap="medium")
    iids = [i["id"] for i in incidents]
    cur = st.session_state.get("sel_incident")
    with left, card("chat"):
        t1, t2 = st.columns([2.2, 1.3], vertical_alignment="center")
        iid = t2.selectbox("Incident", iids, index=iids.index(cur) if cur in iids else 0, label_visibility="collapsed",
                           format_func=lambda x: f"{ui.inc_code(x)} · {code(next(i['model_group'] for i in incidents if i['id'] == x))}",
                           key=f"as_inc_{cur}")
        st.session_state["sel_incident"] = iid
        inc = next(i for i in incidents if i["id"] == iid)
        t1.markdown(f'<div class="sl-h">Incident assistant · {ui.esc(code(inc["model_group"]))}</div>', unsafe_allow_html=True)
        chat = st.session_state.setdefault("chat", {}).setdefault(iid, [])
        pending = st.session_state.pop("ask_pending", None)
        if pending:
            ans = call(c.ask, iid, pending)
            if ans:
                chat.append(ans)
        if not chat:
            html('<div class="sl-help" style="margin:16px 0">Ask a question below, or choose one on the right.</div>')
        for a in chat:
            html(f'<div class="sl-q">{ui.esc(a["question"])}</div>' + answer_html(a))
        with st.form("ask", clear_on_submit=True, border=False):
            q1, q2 = st.columns([5, 1], vertical_alignment="bottom")
            q = q1.text_input("Ask a question", placeholder="Type your question here", max_chars=500)
            if q2.form_submit_button("Send", type="primary", use_container_width=True) and q.strip():
                st.session_state["ask_pending"] = q.strip()
                st.rerun()
        if chat:
            st.button("Clear conversation", key="as_clear", type="tertiary", on_click=lambda: chat.clear())
    with right:
        with card("try"):
            h("Try asking")
            for i, q in enumerate(cached("questions", c.assistant_questions) or []):
                st.button(q, key=f"tq_{i}", use_container_width=True,
                          on_click=lambda q=q: st.session_state.update(ask_pending=q))
        html('<div class="sl-help" style="margin-top:12px">The assistant uses only this incident\'s records and model '
             'results. It says "not determined" when they cannot answer a question.</div>')


def answer_html(a) -> str:
    secs = []
    for s in a["sections"]:
        col = SEC_COLOR.get(s["kind"], "#97a6b7")
        src = SEC_SOURCE.get(s["kind"], "")
        secs.append(f'<div class="sec"><div class="lab" style="color:{col}">{ui.esc(s["title"])}'
                    + (f' <span class="sl-dim" style="text-transform:none;letter-spacing:0">· {ui.esc(src)}</span>' if src else "")
                    + f'</div><div class="txt">{ui.esc(s["text"])}</div></div>')
    return f'<div class="sl-a"><div class="who">Smart Listener</div>{"".join(secs)}</div>'


# ----------------------------------------------------------------------------- 8. settings
def settings_page():
    page_head("Settings", "Administrator account, notifications and the machine registry.")
    c = client()
    sysinfo = system()
    left, right = st.columns(2, gap="medium")
    with left, card("account"):
        h("Administrator account")
        acc = call(c.account) or {}
        html(ui.kv_grid([("Username", ui.esc(acc.get("username", c.username))),
                         ("Last sign-in (PKT)", ui.esc(ui.fmt_time(acc.get("last_login_utc"))) or "-")]))
        with st.form("pw", clear_on_submit=True, border=False):
            cur = st.text_input("Current password", type="password")
            a, b = st.columns(2)
            new = a.text_input("New password", type="password")
            new2 = b.text_input("Repeat new password", type="password")
            if st.form_submit_button("Change password", type="primary"):
                if new != new2:
                    st.error("The new passwords do not match.")
                elif call(c.change_password, cur, new):
                    st.success("Password changed. Other sessions were signed out.")
    mail = call(c.mail_status) or {}
    with right, card("email"):
        t1, t2 = st.columns([3, 1.3], vertical_alignment="center")
        t1.markdown('<div class="sl-h">Email notifications</div>', unsafe_allow_html=True)
        chip = ('<span class="sl-chip steel">SMTP CONFIGURED</span>' if mail.get("mode") == "smtp"
                else '<span class="sl-chip">TEST MODE</span>')
        t2.markdown(f'<div style="text-align:right">{chip}</div>', unsafe_allow_html=True)
        html(f'<div class="sl-help">An e-mail goes to the machine\'s owner when a new HIGH incident opens; repeated HIGH '
             f'windows do not send duplicates. {ui.esc(sysinfo.get("email_note", ""))}</div>')
        s = call(c.settings) or {}
        with st.form("settings", border=False):
            alerts_on = st.checkbox("Send e-mails for new HIGH incidents", bool(s.get("alerts_enabled", True)))
            a, b = st.columns(2)
            per_hour = a.number_input("Most e-mails per hour", 1, 100, int(s.get("max_emails_per_hour", 10)))
            remind = b.number_input("Reminder (minutes, 0 = off)", 0, 1440, int(s.get("reminder_minutes", 0)))
            speed = st.number_input("Simulated monitoring: real seconds per 10-second clip", 0.05, 10.0,
                                    float(s.get("replay_seconds_per_window", 1.0)), step=0.05)
            if st.form_submit_button("Save notification settings", type="primary"):
                if call(c.update_settings, alerts_enabled=alerts_on, max_emails_per_hour=int(per_hour),
                        reminder_minutes=int(remind), replay_seconds_per_window=float(speed)):
                    st.success("Settings saved.")
    html('<div style="height:16px"></div>')
    email_delivery_card(mail)
    html('<div style="height:16px"></div>')
    with card("registry"):
        lim = limits()
        h("Machine registry", "Machine type is derived from the machine ID. Each machine uses its own anomaly model; "
                              "scores are calibrated per machine to shared limits.")
        machines = call(c.machines, True) or []
        models = sysinfo.get("models", {})
        thr = f"{lim[0]:.2f} / {lim[1]:.2f}" if lim else "-"
        w = [0.95, 0.65, 1.1, 1.0, 1.25, 1.0, 1.75, 0.6]
        th(st.columns(w), ["Machine ID", "Type", "Name", "Model", "Limits z (warn / alarm)", "Test ROC-AUC", "Alert e-mail", ""])
        if not machines:
            html('<div class="sl-help">No machines registered.</div>')
        for m in machines:
            g = m["model_group"]
            p = profiles().get(g, {})
            ok = models.get(g, {}).get("available")
            with st.container(key=f"row-reg-{m['id']}"):
                cols = st.columns(w, vertical_alignment="center")
                cell(cols[0], ui.esc(code(g)), True)
                cell(cols[1], ui.esc(ui.machine_type_of(g)))
                cell(cols[2], ui.esc(m["name"]) + ("" if m["active"] else ' <span class="sl-dim">(inactive)</span>'))
                cell(cols[3], f'<span style="color:{ui.SEVERITY["NORMAL"]}">Available</span>' if ok else
                     f'<span style="color:{ui.SEVERITY["HIGH"]}" title="{ui.esc(models.get(g, {}).get("reason", ""))}">Missing</span>')
                cell(cols[4], f'<span style="color:var(--muted)">{thr}</span>')
                cell(cols[5], (f"{p['test_auc']:.3f}" if p.get("test_auc") is not None else "-")
                     + (f' <span style="color:{ui.SEVERITY["WARNING"]}">needs care</span>' if p.get("caution") else ""))
                cell(cols[6], ui.esc(m["owner_email"]))
                cols[7].button("View →", key=f"rv_{m['id']}", type="tertiary", on_click=goto, args=("machines", m["id"]))
    html('<div style="height:16px"></div>')
    with st.expander("Audit trail"):
        rows = call(c.audit, 200) or []
        html(ui.table(["Time (PKT)", "User", "Action", "Detail"],
                      [[ui.esc(ui.fmt_time(r["created_utc"])), ui.esc(r["actor"]), ui.esc(r["action"]), ui.esc(r["detail"])] for r in rows],
                      empty="No audit entries."))


def email_delivery_card(mail: dict):
    c = client()
    with card("delivery"):
        h("E-mail delivery", "Messages waiting, sent and failed. Failed messages are retried automatically up to "
                             f"{mail.get('max_attempts', 3)} times.")
        if mail.get("config_error"):
            html(ui.banner("warn", "<b>The mail settings could not be read:</b> " + ui.esc(mail["config_error"])))
        if mail.get("mode") == "smtp":
            server = f"{mail['host']}:{mail['port']} ({mail['security'].upper()})"
            cfg_rows = [("Mode", "SMTP: messages are sent"), ("Mail server", ui.esc(server)),
                        ("Sender", ui.esc(mail.get("sender") or "-")),
                        ("Password", "set" if mail.get("password_set") else "not set")]
        else:
            cfg_rows = [("Mode", "TEST MODE: messages are saved, not sent"), ("Saved in", ui.esc(mail.get("outbox", ""))),
                        ("To send real e-mails", "fill in config/smtp.env and restart the API")]
        last = mail.get("last_run") or {}
        cfg_rows += [("Automatic sending", f"every {mail.get('interval_s', 30):g} s" if mail.get("worker_running")
                      else "off (use Send pending now)"),
                     ("Last run (PKT)", ui.esc(ui.fmt_time(last.get("time_utc"))) or "not yet")]
        counts = mail.get("counts", {})
        l, r = st.columns([1.2, 1], gap="medium")
        with l:
            html(ui.facts(cfg_rows, left=True))
            if last.get("error"):
                html(ui.banner("warn", "<b>Last run failed:</b> " + ui.esc(last["error"])))
        with r:
            html(ui.stat_strip([("Waiting", str(counts.get("pending", 0)), None), ("Sent", str(counts.get("sent", 0)), None),
                                ("Test mode", str(counts.get("dry_run", 0)), None), ("Failed", str(counts.get("failed", 0)), None)]))
            if st.button("Send pending now", key="mail_now"):
                res = call(c.mail_process)
                if res is not None:
                    st.session_state["mail_flash"] = (
                        f"Sent {res['sent']}, saved in test mode {res['saved_test_mode']}, will retry {res['retry_later']}, "
                        f"failed {res['failed']}, waiting for the hourly limit {res['waiting_for_hourly_limit']}.")
                    st.rerun()
            with st.form("mail_test", clear_on_submit=False, border=False):
                q1, q2 = st.columns([2.2, 1], vertical_alignment="bottom")
                to = q1.text_input("Send a test e-mail to", placeholder="name@example.com")
                if q2.form_submit_button("Send test", use_container_width=True) and to.strip():
                    r_ = call(c.mail_test, to.strip())
                    if r_:
                        st.session_state["mail_flash"] = f"Test e-mail to {r_['recipient']}: " + ui.alert_text(r_) + "."
                        st.rerun()
        if st.session_state.get("mail_flash"):
            html(ui.banner("info", ui.esc(st.session_state.pop("mail_flash"))))
        rows = call(c.alerts, None, 50) or []
        html('<div class="sl-label" style="margin-top:8px">Recent e-mails</div>' +
             ui.table(["Created (PKT)", "Type", "Recipient", "Subject", "Status", "Tries", "Problem"],
                      [[ui.esc(ui.fmt_short(a["created_utc"])), ui.esc(ui.ALERT_KIND.get(a["kind"], a["kind"])),
                        ui.esc(a["recipient"]), ui.esc(a["subject"]), ui.alert_pill(a["status"]), str(a["attempts"]),
                        f'<span class="dim">{ui.esc(a["last_error"])}</span>'] for a in rows],
                      ["high" if a["status"] == "failed" else "" for a in rows], empty="No e-mails recorded yet."))
        failed = [a for a in rows if a["status"] == "failed"]
        if failed:
            f1, f2, _ = st.columns([2, 1, 2], vertical_alignment="bottom")
            pick = f1.selectbox("Failed e-mail", failed, format_func=lambda a: f"#{a['id']} · {a['recipient']} · {a['subject']}")
            if f2.button("Put back in queue", key="mail_retry"):
                if call(c.retry_alert, pick["id"]):
                    st.session_state["mail_flash"] = f"E-mail #{pick['id']} is back in the queue."
                    st.rerun()


# ----------------------------------------------------------------------------- main
PAGES = {"overview": overview_page, "analyze": analyze_page, "machines": machines_page, "incidents": incidents_page,
         "reports": reports_page, "assistant": assistant_page, "settings": settings_page}


def main():
    if st.session_state.get("client") is None:
        login_page()
        return
    c = client()
    page = st.session_state.setdefault("page", "overview")
    sims = running_simulations()
    html(ui.appbar('<span class="sl-chip">SIMULATED MONITORING</span>' if sims else ""))
    with st.sidebar:
        html('<div class="sl-nav-label">Workspace</div>')
        for key, label in WORKSPACE:
            st.button(label, key=f"nav_{key}", type="primary" if page == key else "secondary", use_container_width=True,
                      on_click=goto, args=(key,))
        html('<div class="sl-nav-sep"></div><div class="sl-nav-label">Administration</div>')
        for key, label in ADMIN:
            st.button(label, key=f"nav_{key}", type="primary" if page == key else "secondary", use_container_width=True,
                      on_click=goto, args=(key,))
        html(f'<div class="sl-nav-sep"></div><div class="sl-user">Signed in as {ui.esc(c.username)} · times in PKT (Asia/Karachi)</div>')
        if st.button("Sign out", key="nav_signout", use_container_width=True):
            call(c.logout)
            st.session_state.clear()
            st.rerun()
    alert_banner()
    PAGES.get(page, overview_page)()


main()
