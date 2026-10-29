"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone Project 1

Single-file Streamlit app. No database. Gemini key is entered at runtime
(or via st.secrets["GEMINI_API_KEY"]); without one, the app runs in demo mode.
"""
import json
import re
import urllib.parse
from datetime import datetime

import streamlit as st
from PIL import Image

try:
    import google.generativeai as genai
except ImportError:  # app still runs in demo mode
    genai = None

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Construction Intelligence Platform", page_icon="🏗️", layout="wide")

st.markdown(
    """
<style>
:root { --ink:#1f2a33; --concrete:#eceae6; --steel:#3b4b59; --hivis:#f2b705; --line:#d4d1cb; }
.block-container { padding-top: 1.6rem; max-width: 1200px; }
.hero { background: var(--steel); color:#fff; padding: 1.4rem 1.6rem; border-radius: 6px;
        border-left: 10px solid var(--hivis); margin-bottom: 1rem; }
.hero h1 { margin:0; font-size:1.7rem; color:#fff; }
.hero p { margin:.3rem 0 0; color:#d9e0e6; font-size:.95rem; }
.pill { display:inline-block; padding:.2rem .7rem; border-radius:999px; font-weight:600; font-size:.85rem; }
.pill-ok { background:#dff3e4; color:#1b6b34; } .pill-warn { background:#fff1c9; color:#7a5a00; }
.pill-bad { background:#fde0dc; color:#9b1c10; } .pill-crit { background:#7a0d0d; color:#fff; }
.pill-low { background:#e3eef8; color:#1d4f7a; }
.card { border:1px solid var(--line); border-radius:6px; padding:.9rem 1rem; margin-bottom:.6rem; background:#fff; }
.card h4 { margin:0 0 .3rem; font-size:1rem; }
.alert { border-left:6px solid var(--hivis); background:#fff8e1; padding:.8rem 1rem; border-radius:4px; margin:.4rem 0; }
.alert.red { border-color:#c0392b; background:#fdecea; }
.demo-banner { background:#fff1c9; color:#6b5000; padding:.5rem .8rem; border-radius:4px; font-size:.9rem; margin-bottom:.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
MAX_IMAGE_MB = 8
SEVERITY_CLASS = {"Low": "pill-low", "Medium": "pill-warn", "High": "pill-bad", "Critical": "pill-crit"}

# Indicative per-sqft construction cost (INR) by structure type and city tier
BUILD_COST = {
    "Residential (RCC Framed)": {"Tier 1": 2400, "Tier 2": 2000, "Tier 3": 1700},
    "Commercial Office": {"Tier 1": 3200, "Tier 2": 2700, "Tier 3": 2300},
    "Industrial Shed (PEB)": {"Tier 1": 1900, "Tier 2": 1650, "Tier 3": 1450},
    "Luxury Villa": {"Tier 1": 4200, "Tier 2": 3500, "Tier 3": 3000},
}
# Indicative completed-asset market value per sqft (INR)
MARKET_RATE = {
    "Residential (RCC Framed)": {"Tier 1": 9500, "Tier 2": 5200, "Tier 3": 3300},
    "Commercial Office": {"Tier 1": 14000, "Tier 2": 7500, "Tier 3": 4500},
    "Industrial Shed (PEB)": {"Tier 1": 5000, "Tier 2": 3200, "Tier 3": 2200},
    "Luxury Villa": {"Tier 1": 16000, "Tier 2": 8500, "Tier 3": 5000},
}
# kg steel and 50-kg cement bags per sqft (rule-of-thumb)
MATERIAL_RATE = {
    "Residential (RCC Framed)": (4.0, 0.42),
    "Commercial Office": (5.5, 0.50),
    "Industrial Shed (PEB)": (9.0, 0.25),
    "Luxury Villa": (4.8, 0.48),
}
MILESTONES = [(0, "Site prep"), (10, "Foundation"), (30, "Plinth & columns"), (55, "Slabs & structure"),
              (75, "Masonry & services"), (90, "Finishing"), (100, "Handover")]

WEATHER_ALERTS = {
    "High Wind": [
        ("red", "Suspend crane and hoisting operations when sustained wind exceeds ~36 km/h (OSHA 1926.1431 guidance); follow the manufacturer's limit if lower."),
        ("red", "Stop work on scaffolds, formwork and roof edges. Secure sheets, loose materials and tower crane slewing brakes."),
        ("", "Inspect scaffold ties, bracing and hoarding before restarting work.")],
    "Heavy Rain": [
        ("red", "Electrical hazard: de-energise temporary power, use RCD/GFCI-protected circuits, and keep cable joints off wet ground."),
        ("", "Concrete: stop pours; cover fresh concrete with tarpaulin to prevent surface washout and check slump on restart."),
        ("", "Excavations: inspect for water ingress, dewater, and check shoring before anyone enters."),
        ("", "Anti-skid walkways; restrict work at height until surfaces are dry.")],
    "Extreme Heat": [
        ("red", "Heat-stress control: scheduled water/shade breaks, avoid heavy work 12:00-15:30, monitor for heat exhaustion."),
        ("", "Concrete curing: pour early morning or at night, use chilled mixing water and retarders, and begin wet curing for at least 7 days (IS 456 cl. 13.5)."),
        ("", "Keep hot-work and fuel storage shaded; check steel surfaces for burn risk.")],
    "Normal": [
        ("", "Normal operations. Hold the daily toolbox talk and complete the pre-task hazard checklist."),
        ("", "Verify PPE, housekeeping, and the lifting plan before shift start.")],
}

SAFETY_PROMPT = """You are a certified construction safety auditor (OSHA 29 CFR 1926 and Indian IS 456 / BOCW site practice).
Analyse the construction-site photo for worker PPE compliance (helmets, high-visibility vests, steel-toe boots, safety harnesses
where work at height is visible) and other site hazards. Be conservative: use "Not Visible" if you cannot tell.
Return ONLY JSON with this schema:
{"safety_score": int 0-100, "overall_status": "Compliant"|"Needs Improvement"|"Non-Compliant",
 "summary": str, "workers_detected": int,
 "ppe_detection": {"helmet": "Compliant"|"Violation"|"Not Visible", "high_vis_vest": same, "steel_toe_boots": same, "safety_harness": same},
 "hazards": [{"hazard": str, "severity": "Low"|"Medium"|"High"|"Critical", "standard": str, "remedy": str}]}"""

CRACK_PROMPT = """You are a structural engineer diagnosing defects in concrete/masonry surfaces (IS 456, IS 13935, IS 15988).
Classify the visible defect. Return ONLY JSON with this schema:
{"defect_type": "Hairline"|"Settlement"|"Structural Shear"|"Thermal Expansion"|"Spalling",
 "estimated_width_mm": float, "pattern": str, "confidence_pct": int 0-100,
 "severity": "Low"|"Medium"|"High"|"Critical", "urgency": "Monitor"|"Repair within 3 months"|"Repair within 2 weeks"|"Immediate action",
 "root_causes": [str], "recommended_repairs": [str], "monitoring_advice": str, "needs_structural_engineer": bool}
State that the width is a visual estimate."""

ASSISTANT_SYSTEM = """You are a senior site engineer assisting on an Indian construction project. Give practical, concise answers citing
IS codes (IS 456, IS 800, IS 1893, NBC 2016) and OSHA where relevant. Use the current weather context. Flag when a licensed
structural engineer must be consulted. Do not invent code clause numbers you are unsure of."""

# ----------------------------------------------------------------------------
# Demo-mode fixtures
# ----------------------------------------------------------------------------
DEMO_SAFETY = {
    "safety_score": 62, "overall_status": "Needs Improvement", "workers_detected": 4,
    "summary": "DEMO DATA: Most workers wear helmets, but vest and harness compliance is inconsistent near the edge.",
    "ppe_detection": {"helmet": "Compliant", "high_vis_vest": "Violation", "steel_toe_boots": "Not Visible", "safety_harness": "Violation"},
    "hazards": [
        {"hazard": "Worker at slab edge without harness", "severity": "Critical", "standard": "OSHA 1926.501", "remedy": "Stop work, install edge protection and anchor points, issue harnesses."},
        {"hazard": "Loose material stacked near edge", "severity": "High", "standard": "OSHA 1926.250", "remedy": "Relocate stacks and install toe boards."},
        {"hazard": "Poor housekeeping on walkway", "severity": "Low", "standard": "OSHA 1926.25", "remedy": "Clear debris; assign housekeeping crew."}],
}
DEMO_CRACK = {
    "defect_type": "Settlement", "estimated_width_mm": 3.5, "pattern": "Diagonal stair-step crack widening toward the top",
    "confidence_pct": 78, "severity": "High", "urgency": "Repair within 2 weeks",
    "root_causes": ["Differential foundation settlement", "Poor soil compaction or leaking drain near footing", "Overloading of one wall panel"],
    "recommended_repairs": ["Install crack gauges and monitor for 2-4 weeks", "Underpin or stabilise foundation if movement continues",
                            "Epoxy injection (low-viscosity) for active-free cracks", "Structural re-grouting with non-shrink grout, then re-plaster with mesh"],
    "monitoring_advice": "Date-mark tell-tales and re-measure weekly. DEMO DATA: visual estimate only.", "needs_structural_engineer": True,
}


def demo_chat_reply(question: str, weather: str) -> str:
    q = question.lower()
    if any(k in q for k in ["curing", "cure"]):
        return "Cure for at least 7 days for OPC (10 days for blended cement or in hot weather), keeping surfaces continuously moist (IS 456). *(Demo mode)*"
    if any(k in q for k in ["crack", "settlement"]):
        return "Measure width and track it over weeks. Hairline cracks under 0.3 mm are usually cosmetic; widening or diagonal cracks need an engineer. *(Demo mode)*"
    if any(k in q for k in ["ppe", "helmet", "harness"]):
        return "Minimum PPE: helmet, high-vis vest, safety boots; add a full-body harness for work above 1.8 m (OSHA) with a rescue plan. *(Demo mode)*"
    if any(k in q for k in ["cement", "steel", "material"]):
        return "Use the Planning tab for steel/cement estimates; as a rule of thumb about 4 kg steel and 0.4 bag cement per sqft for residential RCC. *(Demo mode)*"
    return f"Demo mode reply. Current weather is **{weather}**; check the advisory above before assigning today's tasks. Add a Gemini API key for full answers."


# ----------------------------------------------------------------------------
# Gemini helpers
# ----------------------------------------------------------------------------
def get_api_key() -> str:
    key = st.session_state.get("api_key", "").strip()
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            key = ""
    return key


def live_mode() -> bool:
    return bool(get_api_key()) and genai is not None


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def gemini_json(prompt: str, image: Image.Image) -> dict:
    genai.configure(api_key=get_api_key())
    model = genai.GenerativeModel(
        st.session_state.get("model", MODELS[0]),
        generation_config={"response_mime_type": "application/json", "temperature": 0.2},
    )
    return parse_json(model.generate_content([prompt, image]).text)


def gemini_text(prompt: str) -> str:
    genai.configure(api_key=get_api_key())
    model = genai.GenerativeModel(st.session_state.get("model", MODELS[0]), generation_config={"temperature": 0.4})
    return model.generate_content(prompt).text


def load_image(uploaded) -> Image.Image | None:
    if uploaded.size > MAX_IMAGE_MB * 1024 * 1024:
        st.error(f"Image is larger than {MAX_IMAGE_MB} MB. Upload a smaller file.")
        return None
    try:
        img = Image.open(uploaded)
        img.load()
        img = img.convert("RGB")
        img.thumbnail((1600, 1600))
        return img
    except Exception:
        st.error("Could not read this image. Upload a valid JPG or PNG.")
        return None


def analyse(prompt: str, img: Image.Image, demo: dict) -> dict | None:
    if not live_mode():
        return demo
    try:
        with st.spinner("Analysing image with Gemini..."):
            return gemini_json(prompt, img)
    except Exception as exc:
        st.error(f"Gemini request failed: {exc}. Check the API key, model, and quota.")
        return None


def pill(text: str, cls: str) -> str:
    return f'<span class="pill {cls}">{text}</span>'


def download_report(data: dict, name: str):
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "mode": "live" if live_mode() else "demo", **data}
    st.download_button("Download JSON report", json.dumps(payload, indent=2),
                       file_name=f"{name}_{datetime.now():%Y%m%d_%H%M}.json", mime="application/json")


def fmt_inr(amount: float) -> str:
    if amount >= 1e7:
        return f"₹{amount / 1e7:,.2f} Cr"
    return f"₹{amount / 1e5:,.2f} Lakhs"


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
def tab_safety():
    st.subheader("Site safety and PPE audit")
    up = st.file_uploader("Upload a construction site photo", type=["jpg", "jpeg", "png"], key="safety_up")
    if not up:
        st.info("Upload a site photo to audit PPE compliance and hazards.")
        return
    img = load_image(up)
    if img is None:
        return
    left, right = st.columns([1, 1.3])
    left.image(img, caption="Uploaded site photo", use_container_width=True)
    if not right.button("Run safety audit", type="primary", key="run_safety"):
        right.caption("Press the button to start the audit.")
        return
    result = analyse(SAFETY_PROMPT, img, DEMO_SAFETY)
    if not result:
        return
    st.session_state["safety_result"] = result


def render_safety(result: dict):
    score = int(result.get("safety_score", 0))
    status = result.get("overall_status", "Unknown")
    cls = "pill-ok" if status == "Compliant" else "pill-warn" if status == "Needs Improvement" else "pill-bad"
    c1, c2, c3 = st.columns(3)
    c1.metric("Safety score", f"{score}/100")
    c2.metric("Workers detected", result.get("workers_detected", "n/a"))
    c3.markdown("**Overall status**<br>" + pill(status, cls), unsafe_allow_html=True)
    st.progress(min(max(score, 0), 100) / 100)
    st.write(result.get("summary", ""))

    st.markdown("##### PPE checklist")
    ppe = result.get("ppe_detection", {})
    cols = st.columns(4)
    labels = {"helmet": "Helmet", "high_vis_vest": "High-vis vest", "steel_toe_boots": "Steel-toe boots", "safety_harness": "Safety harness"}
    for col, (k, label) in zip(cols, labels.items()):
        v = ppe.get(k, "Not Visible")
        col.markdown(f"**{label}**<br>" + pill(v, {"Compliant": "pill-ok", "Violation": "pill-bad"}.get(v, "pill-low")), unsafe_allow_html=True)

    st.markdown("##### Identified hazards")
    for h in sorted(result.get("hazards", []), key=lambda x: ["Critical", "High", "Medium", "Low"].index(x.get("severity", "Low")) if x.get("severity") in SEVERITY_CLASS else 9):
        sev = h.get("severity", "Low")
        st.markdown(
            f'<div class="card"><h4>{h.get("hazard","")} &nbsp;{pill(sev, SEVERITY_CLASS.get(sev, "pill-low"))}</h4>'
            f'<small>{h.get("standard","")}</small><br><b>Remedy:</b> {h.get("remedy","")}</div>',
            unsafe_allow_html=True)
    download_report(result, "safety_audit")


def tab_cracks():
    st.subheader("Crack diagnostics and structural integrity")
    up = st.file_uploader("Upload a concrete or wall surface photo", type=["jpg", "jpeg", "png"], key="crack_up")
    if not up:
        st.info("Upload a close-up of the defect. Include a coin or ruler for scale if possible.")
        return
    img = load_image(up)
    if img is None:
        return
    left, right = st.columns([1, 1.3])
    left.image(img, caption="Surface photo", use_container_width=True)
    if right.button("Diagnose defect", type="primary", key="run_crack"):
        result = analyse(CRACK_PROMPT, img, DEMO_CRACK)
        if result:
            st.session_state["crack_result"] = result
    else:
        right.caption("Press the button to start the diagnosis.")


def render_crack(r: dict):
    sev = r.get("severity", "Low")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Defect type", r.get("defect_type", "n/a"))
    c2.metric("Est. crack width", f"{r.get('estimated_width_mm', 0)} mm")
    c3.metric("Confidence", f"{r.get('confidence_pct', 0)}%")
    c4.markdown("**Severity**<br>" + pill(sev, SEVERITY_CLASS.get(sev, "pill-low")), unsafe_allow_html=True)
    st.markdown(f'<div class="alert{" red" if sev in ("High", "Critical") else ""}"><b>Urgency:</b> {r.get("urgency", "n/a")}'
                f'{" · Consult a licensed structural engineer." if r.get("needs_structural_engineer") else ""}</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("##### Likely root causes")
        for x in r.get("root_causes", []):
            st.markdown(f"- {x}")
        st.caption(f"Pattern: {r.get('pattern', 'n/a')}")
    with b:
        st.markdown("##### Recommended repairs")
        for x in r.get("recommended_repairs", []):
            st.markdown(f"- {x}")
    st.info(r.get("monitoring_advice", ""))
    download_report(r, "crack_diagnosis")


def tab_planning():
    st.subheader("Planning and real estate valuation")
    c1, c2 = st.columns(2)
    stype = c1.selectbox("Structure type", list(BUILD_COST))
    tier = c2.selectbox("Location tier", ["Tier 1", "Tier 2", "Tier 3"],
                        help="Tier 1: Mumbai, Delhi, Bengaluru. Tier 2: Bhopal, Indore, Pune. Tier 3: smaller cities.")
    c3, c4 = st.columns(2)
    sqft = c3.number_input("Built-up area (sqft)", 200, 500000, 2000, step=100)
    pct = c4.slider("Completion (%)", 0, 100, 40)

    budget = sqft * BUILD_COST[stype][tier]
    capex = budget * pct / 100
    full_value = sqft * MARKET_RATE[stype][tier]
    # Asset value rises with completion: land/approvals share ~35%, rest follows build progress
    asset_value = full_value * (0.35 + 0.65 * pct / 100) if pct > 0 else full_value * 0.35
    steel_kg, bags = MATERIAL_RATE[stype]
    steel_t, cement = sqft * steel_kg / 1000, sqft * bags

    m = st.columns(4)
    m[0].metric("Total project budget", fmt_inr(budget))
    m[1].metric("Capex incurred", fmt_inr(capex))
    m[2].metric("Current asset valuation", fmt_inr(asset_value), f"Completed: {fmt_inr(full_value)}", delta_color="off")
    m[3].metric("Budget remaining", fmt_inr(budget - capex))
    m2 = st.columns(4)
    m2[0].metric("Steel required", f"{steel_t:,.1f} tonnes")
    m2[1].metric("Steel consumed (est.)", f"{steel_t * pct / 100:,.1f} t")
    m2[2].metric("Cement required", f"{cement:,.0f} bags")
    m2[3].metric("Cement consumed (est.)", f"{cement * pct / 100:,.0f} bags")

    current = [name for p, name in MILESTONES if pct >= p][-1]
    nxt = next(((p, n) for p, n in MILESTONES if p > pct), None)
    st.markdown("##### Milestone progress")
    st.progress(pct / 100, text=f"{pct}% complete · current phase: {current}" + (f" · next: {nxt[1]} at {nxt[0]}%" if nxt else ""))
    st.caption("Estimates use indicative market rates and rule-of-thumb material factors. Replace with local quotations before decisions.")

    st.markdown("##### Conceptual 3D architectural rendering")
    style = st.selectbox("Rendering style", ["Photorealistic 3D render", "Architectural watercolour", "Modern minimalist concept", "Night-time exterior"])
    floors = st.slider("Number of floors", 1, 20, 2)
    extra = st.text_input("Extra design notes", "sustainable materials, rooftop solar, landscaped entrance")
    prompt = f"{style} of a {floors}-storey {stype.split(' (')[0].lower()} building, Indian architecture, {tier} city, {extra}, golden hour, high detail"
    st.code(prompt, language=None)
    if st.button("Generate concept image", key="gen_img"):
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width=1024&height=640&nologo=true"
        st.image(url, caption="Concept rendering via Pollinations AI (illustrative only)", use_container_width=True)


def tab_assistant():
    st.subheader("Daily site assistant and weather-aware advisory")
    weather = st.selectbox("Today's weather condition", list(WEATHER_ALERTS), index=3)
    for kind, msg in WEATHER_ALERTS[weather]:
        st.markdown(f'<div class="alert {kind}">{msg}</div>', unsafe_allow_html=True)

    st.markdown("##### Ask the site engineer")
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": "Hello! Ask me about curing, formwork, PPE, scheduling, or materials."}]
    box = st.container(height=340)
    for msg in st.session_state.chat:
        box.chat_message(msg["role"]).write(msg["content"])

    if st.button("Clear chat", key="clear_chat"):
        st.session_state.pop("chat")
        st.rerun()
    question = st.chat_input("Ask a construction question", key="chat_in")
    if not question:
        return
    st.session_state.chat.append({"role": "user", "content": question})
    if live_mode():
        history = "\n".join(f"{m['role'].title()}: {m['content']}" for m in st.session_state.chat[-10:])
        prompt = f"{ASSISTANT_SYSTEM}\nToday's weather: {weather}.\n\n{history}\nAssistant:"
        try:
            reply = gemini_text(prompt)
        except Exception as exc:
            reply = f"Gemini request failed: {exc}"
    else:
        reply = demo_chat_reply(question, weather)
    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.rerun()


# ----------------------------------------------------------------------------
# Sidebar & main
# ----------------------------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.header("🏗️ Control panel")
        st.text_input("Google Gemini API key", type="password", key="api_key",
                      help="Held in session memory only; never stored. Leave blank for demo mode.")
        st.selectbox("Gemini model", MODELS, key="model")
        if live_mode():
            st.success("Live mode: Gemini connected")
        elif genai is None:
            st.warning("google-generativeai not installed. Running in demo mode.")
        else:
            st.warning("Demo mode: sample outputs only")
        st.divider()
        st.markdown("**Features**\n- Safety and PPE audit\n- Crack diagnostics\n- Planning and valuation\n- Weather advisory and assistant")
        with st.expander("Architecture overview"):
            st.markdown(
                "- **UI:** Streamlit single-page app with tabs\n"
                "- **Vision/LLM:** Gemini multimodal, JSON-constrained output\n"
                "- **Calculations:** Local rate tables, no external services\n"
                "- **Renders:** Pollinations AI image URL\n"
                "- **State:** `st.session_state` only, no database")
        st.caption("AI findings are decision support, not a substitute for a certified safety officer or structural engineer.")


def main():
    sidebar()
    st.markdown('<div class="hero"><h1>Multimodal Construction Intelligence Platform</h1>'
                '<p>Safety audits, structural diagnostics, project valuation and a weather-aware site assistant.</p></div>',
                unsafe_allow_html=True)
    if not live_mode():
        st.markdown('<div class="demo-banner">Demo mode: image analysis returns sample results. Add a Gemini API key in the sidebar for live analysis.</div>',
                    unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Safety and PPE audit", "Crack diagnostics", "Planning and valuation", "Site assistant"])
    with t1:
        tab_safety()
        if "safety_result" in st.session_state:
            render_safety(st.session_state["safety_result"])
    with t2:
        tab_cracks()
        if "crack_result" in st.session_state:
            render_crack(st.session_state["crack_result"])
    with t3:
        tab_planning()
    with t4:
        tab_assistant()


if __name__ == "__main__":
    main()
