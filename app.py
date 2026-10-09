import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import base64
import os
from sklearn.metrics import mean_squared_error, r2_score

# ══════════════════════════════════════════════
#  PAGE CONFIG  — must be first Streamlit call
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Abaca Girth Prediction System",
    page_icon="🌴",
    layout="wide",
)

# ══════════════════════════════════════════════
#  BACKGROUND + GLOBAL CSS
# ══════════════════════════════════════════════
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_css = ""
if os.path.exists("background.jpg"):
    bg_b64 = get_base64("background.jpg")
    bg_css = f"""
    .stApp {{
        background: linear-gradient(rgba(0,20,5,0.72), rgba(5,30,10,0.68)),
                    url(data:image/jpg;base64,{bg_b64});
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}"""
else:
    bg_css = ".stApp { background: linear-gradient(160deg,#0a1f0d,#163d20,#0b1e0f); }"

st.markdown(f"""
<style>
{bg_css}

/* ── Reset & base ────────────────────────── */
* {{ box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family: 'Trebuchet MS', sans-serif; }}
section.main > div {{ padding-top: 0 !important; padding-bottom: 0 !important; }}
.block-container {{ padding: 0.6rem 1.2rem 0.4rem 1.2rem !important; max-width: 100% !important; }}

/* hide default streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── All text white ──────────────────────── */
h1,h2,h3,h4,h5,h6,p,label,span,div,
[data-testid="stMarkdownContainer"] p {{ color: #e8f5e1 !important; }}

/* ── Section header pills ────────────────── */
.sec-header {{
    background: rgba(20,80,35,0.80);
    border: 1.5px solid #4ade80;
    border-radius: 8px;
    padding: 5px 14px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #c6ffda !important;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 10px;
}}

/* ── Panel cards ─────────────────────────── */
.panel {{
    background: rgba(5,35,12,0.60);
    border: 1px solid rgba(74,222,128,0.30);
    border-radius: 14px;
    padding: 12px 16px;
    height: 100%;
}}

/* ── Slider label ──────────────────────────── */
.slider-label {{
    font-size: 0.78rem;
    color: #b0dbb8 !important;
    margin-bottom: 0;
    line-height: 1.2;
}}

/* ── Slider track & thumb ────────────────── */
.stSlider [data-baseweb="slider"] {{
    padding: 0 !important;
}}
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: #4ade80 !important;
    border: 2px solid #166534 !important;
}}
.stSlider [data-baseweb="slider"] div[class*="Track"] {{
    background: rgba(74,222,128,0.20) !important;
}}

/* ── Result box ──────────────────────────── */
.result-box {{
    background: rgba(22,101,52,0.45);
    border: 2px solid #4ade80;
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center;
    margin: 8px 0;
}}
.result-girth {{
    font-size: 2.6rem;
    font-weight: 800;
    color: #86efac !important;
    line-height: 1.1;
}}
.result-unit {{ font-size: 0.9rem; color: #a7f3d0 !important; }}

/* ── Metric cards ────────────────────────── */
[data-testid="metric-container"] {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(74,222,128,0.30) !important;
    border-radius: 10px !important;
    padding: 8px 10px !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #86efac !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.72rem !important;
    color: #a7f3d0 !important;
}}

/* ── Title banner ────────────────────────── */
.title-banner {{
    background: rgba(10,50,20,0.75);
    border: 1.5px solid rgba(74,222,128,0.50);
    border-radius: 12px;
    padding: 7px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}}
.title-text {{
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    color: #d1fae5 !important;
    text-transform: uppercase;
}}
.title-sub {{
    font-size: 0.72rem;
    color: #86efac !important;
    letter-spacing: 0.04em;
}}

/* ── Flag styles inside banner ───────────── */
.flag-wrap {{
    margin-left: auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
    flex-shrink: 0;
}}
.flag-wrap img {{
    height: 42px;
    width: 70px;
    object-fit: cover;
    border-radius: 5px;
    border: 1.5px solid rgba(255,255,255,0.30);
    box-shadow: 0 2px 8px rgba(0,0,0,0.50);
}}
.flag-label {{
    font-size: 0.58rem !important;
    color: #c6ffda !important;
    letter-spacing: 0.07em;
    opacity: 0.85;
    text-transform: uppercase;
}}

/* ── Dividers ─────────────────────────────── */
hr {{ border-color: rgba(74,222,128,0.20) !important; margin: 6px 0 !important; }}

/* ── pyplot transparent ──────────────────── */
.stPlotlyChart, .stPyplot {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════
@st.cache_resource
def load_model():
    with open("abaca_rf_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    pkg          = load_model()
    model        = pkg["model"]
    feature_cols = pkg["features"]   # ['height_cm','leaf_count','moisture_pct','soil_pH','temperature','humidity_pct','sun_shade_pct']
    data_ranges  = pkg.get("data_ranges", {})
except FileNotFoundError:
    st.error("❌ `abaca_rf_model.pkl` not found. Place it in the same folder as this script.")
    st.stop()

# ── Hardcoded cross-val metrics from your RF training output ──
R2_VAL   = 0.8664
RMSE_VAL = 1.3598
MAE_VAL  = 2.0324

# ══════════════════════════════════════════════
#  HEADER  (logo + title + PH flag)
# ══════════════════════════════════════════════
logo_html = ""
if os.path.exists("logo.png"):
    logo_b64  = get_base64("logo.png")
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:62px;width:62px;object-fit:contain;border-radius:50%;">'

# Philippine flag — embedded as base64 so no external file needed
if os.path.exists("PHflag.png"):
    PHflag_b64  = get_base64("PHflag.png")
    PHflag_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:62px;width:62px;object-fit:contain;border-radius:50%;">'

st.markdown(f"""
<div class="title-banner">
    {logo_html}
    <div>
        <div class="title-text">🌴 Abaca Girth Prediction System</div>
        <div class="title-sub">Machine Learning–Based Agricultural Prediction Tool · Random Forest Regressor · San Rafael National High School</div>
    </div>
    <div class="flag-wrap">
        <img src="data:image/png;base64,{PHflag_b64}" alt="Philippine Flag">
        <span class="flag-label">Philippines</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  INIT SESSION STATE
#  Keys now match feature_cols exactly
# ══════════════════════════════════════════════
defaults = {
    "height_cm":      178.0,
    "leaf_count":       4.0,
    "moisture_pct":    96.0,
    "soil_pH":          5.67,
    "temperature":     29.9,
    "humidity_pct":    67.0,
    "sun_shade_pct":   58.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════
#  PREDICTION — NO log transform (model trained on raw girth)
# ══════════════════════════════════════════════
def run_prediction():
    input_df = pd.DataFrame([[
        st.session_state["height_cm"],
        st.session_state["leaf_count"],
        st.session_state["moisture_pct"],
        st.session_state["soil_pH"],
        st.session_state["temperature"],
        st.session_state["humidity_pct"],
        st.session_state["sun_shade_pct"],
    ]], columns=feature_cols)

    pred_girth = max(float(model.predict(input_df)[0]), 0.5)
    st.session_state["last_pred"] = pred_girth

def on_slider_change(key):
    st.session_state[key] = st.session_state[f"sl_{key}"]
    run_prediction()

if "last_pred" not in st.session_state:
    run_prediction()

# ══════════════════════════════════════════════
#  3-COLUMN LAYOUT
# ══════════════════════════════════════════════
col_left, col_mid, col_right = st.columns([1.05, 1.0, 1.0], gap="medium")

# ─────────────────────────────────────────────
#  LEFT — INPUT PARAMETERS
# ─────────────────────────────────────────────
with col_left:
    st.markdown('<div class="sec-header">⬇ Input Parameters</div>', unsafe_allow_html=True)

    params = [
        ("height_cm",     "Plant Height (cm)",  75.0,  240.0, 1.0,  "%.0f"),
        ("leaf_count",    "Leaf Count",           1.0,   10.0, 1.0,  "%.0f"),
        ("moisture_pct",  "Soil Moisture (%)",   87.0,   99.0, 0.5,  "%.1f"),
        ("soil_pH",       "Soil pH",              3.5,    9.0, 0.01, "%.2f"),
        ("temperature",   "Temperature (°C)",    25.0,   35.0, 0.1,  "%.1f"),
        ("humidity_pct",  "Humidity (%)",        56.0,   73.0, 1.0,  "%.0f"),
        ("sun_shade_pct", "Sunshade Index (%)",  44.0,   69.0, 1.0,  "%.0f"),
    ]

    for key, label, lo, hi, step, fmt in params:
        st.markdown(f'<div class="slider-label">{label}</div>', unsafe_allow_html=True)
        st.slider(
            label, lo, hi,
            value=float(st.session_state[key]),
            step=step, format=fmt,
            key=f"sl_{key}",
            on_change=on_slider_change, args=(key,),
            label_visibility="collapsed",
        )

# ─────────────────────────────────────────────
#  MIDDLE — PREDICTION + FEATURE IMPORTANCE
# ─────────────────────────────────────────────
with col_mid:
    st.markdown('<div class="sec-header">🎯 Prediction</div>', unsafe_allow_html=True)

    pred_girth = st.session_state.get("last_pred", 0.0)
    st.markdown(f"""
    <div class="result-box">
        <div class="result-unit" style="margin-bottom:2px;">Predicted Girth</div>
        <div class="result-girth">{pred_girth:.2f}</div>
        <div class="result-unit">centimeters</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="sec-header" style="margin-top:4px;">📊 Feature Importance (%) — Random Forest</div>', unsafe_allow_html=True)

    imps     = model.feature_importances_
    imps_pct = (imps / imps.sum()) * 100
    idx      = np.argsort(imps_pct)
    fsorted  = np.array(feature_cols)[idx]
    vsorted  = imps_pct[idx]

    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    colors = ["#22c55e" if v == vsorted.max() else "#166534" for v in vsorted]
    bars   = ax.barh(fsorted, vsorted, color=colors, edgecolor="none", height=0.55)

    for bar, val in zip(bars, vsorted):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", color="#d1fae5", fontsize=7.5)

    ax.set_xlabel("Percentage", color="#a7f3d0", fontsize=7.5)
    ax.tick_params(colors="#c6ffda", labelsize=7.5)
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color((0.29, 0.87, 0.50, 0.25))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="x", linestyle="--", alpha=0.18, color="#4ade80")
    ax.set_xlim(0, vsorted.max() * 1.22)
    plt.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────────────
#  RIGHT — MODEL ACCURACY METRICS
# ─────────────────────────────────────────────
with col_right:
    st.markdown('<div class="sec-header">✅ Model Accuracy Metrics</div>', unsafe_allow_html=True)

    st.metric("Mean Absolute Error",      f"{MAE_VAL:.4f}")
    st.metric("Root Mean Square Error", f"{RMSE_VAL:.4f}")
    st.metric("R² Score",               f"{R2_VAL:.4f}", help="Closer to 1.0 = better fit")

    st.markdown("<hr>", unsafe_allow_html=True)

    r2_pct = min(max(R2_VAL * 100, 0), 100)
    if R2_VAL >= 0.85:
        gauge_color, gauge_label = "#22c55e", "Excellent"
    elif R2_VAL >= 0.70:
        gauge_color, gauge_label = "#fbbf24", "Good"
    else:
        gauge_color, gauge_label = "#f87171", "Fair"

    st.markdown(f"""
    <div style="background:rgba(5,35,12,0.50);border:1px solid rgba(74,222,128,0.25);
                border-radius:12px;padding:10px 14px;margin-top:4px;">
        <div style="font-size:0.72rem;color:#a7f3d0 !important;margin-bottom:6px;">
            Model fit quality
        </div>
        <div style="background:rgba(255,255,255,0.08);border-radius:99px;height:10px;overflow:hidden;">
            <div style="width:{r2_pct:.1f}%;height:100%;background:{gauge_color};border-radius:99px;
                        transition:width 0.6s ease;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:4px;">
            <span style="font-size:0.72rem;color:#6b7280 !important;">0%</span>
            <span style="font-size:0.78rem;font-weight:700;color:{gauge_color} !important;">
                {gauge_label} · {r2_pct:.1f}%
            </span>
            <span style="font-size:0.72rem;color:#6b7280 !important;">100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.72rem;color:#a7f3d0 !important;margin-bottom:6px;">Current input summary</div>', unsafe_allow_html=True)
    summary = {
        "height_cm":      st.session_state["height_cm"],
        "leaf_count":     st.session_state["leaf_count"],
        "moisture_pct":   st.session_state["moisture_pct"],
        "soil_pH":        st.session_state["soil_pH"],
        "temperature":    st.session_state["temperature"],
        "humidity_pct":   st.session_state["humidity_pct"],
        "sun_shade_pct":  st.session_state["sun_shade_pct"],
    }
    rows = "".join([
        f"""<tr>
            <td style='font-size:0.72rem;color:#a7f3d0;padding:2px 6px;'>{k}</td>
            <td style='font-size:0.72rem;color:#d1fae5;padding:2px 6px;text-align:right;
                       font-weight:600;'>{v:.2f}</td>
        </tr>"""
        for k, v in summary.items()
    ])
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;
                  background:rgba(5,35,12,0.40);border-radius:8px;overflow:hidden;">
        {rows}
    </table>
    """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:6px;font-size:0.68rem;color:#4ade80 !important;opacity:0.7;">
    © 2026 · Abaca Girth Prediction System 🌾 &nbsp;|&nbsp; By: Wilmer C. Valencia &nbsp;|&nbsp;
    San Rafael National High School, Tigaon, Camarines Sur
</div>
""", unsafe_allow_html=True)
