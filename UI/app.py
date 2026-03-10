"""
Privacy-Preserving Biometric Authentication System
Streamlit UI — uses EXACT code from the notebook (TenSEAL CKKS)
"""

import streamlit as st
import json
import numpy as np
import time
import io
import base64
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Custom JSON encoder to handle numpy types ──────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, (np.ndarray,)):  return obj.tolist()
        return super().default(obj)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioAuth XAI",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── BIOLUMINESCENT DEEP OCEAN / IRIS SCANNER THEME (REDUCED GLOW) ─────────────────────────────────
st.markdown("""
<style>
/* === DEEP OCEAN BIOLUMINESCENT THEME - REDUCED GLOW === */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* Global Reset - Deep Ocean Parchment */
html, body, [class*="css"] {
    font-family: 'Cormorant Garamond', serif !important;
    background: #0A0C0E !important;
    color: #E8D3B0 !important;
    font-size: 17px !important;
    font-weight: 400 !important;
    letter-spacing: 0.02em !important;
    cursor: default !important;
}
.stApp {
    background: #0A0C0E !important;
    background-image: 
        radial-gradient(circle at 20% 30%, rgba(255, 191, 71, 0.02) 0%, transparent 30%),
        radial-gradient(circle at 80% 70%, rgba(255, 159, 28, 0.02) 0%, transparent 40%) !important;
}
.block-container { 
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    position: relative !important;
    z-index: 2 !important;
}

/* === CURSOR STYLES - ENHANCED VISIBILITY === */
a, button, .stButton > button, [role="button"], 
input, select, textarea, [tabindex]:not([tabindex="-1"]),
.stRadio label, .stCheckbox label, .stSelectbox, 
[data-testid="stFileUploader"], .badge, .crypto-chip,
.streamlit-expanderHeader, .stTabs [data-baseweb="tab"] {
    cursor: pointer !important;
}

.stTextInput input, .stTextArea textarea, .stNumberInput input {
    cursor: text !important;
}

.stProgress, .feat-track, .stProgress > div {
    cursor: default !important;
}

/* === IRIS RINGS - BIOLUMINESCENT ANIMATION (SCAN LINE REMOVED) === */
.iris-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
    overflow: hidden;
}
.iris-ring-outer {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 80vmin;
    height: 80vmin;
    border: 2px solid rgba(255, 191, 71, 0.1);
    border-radius: 50%;
    animation: irisPulse 12s ease-in-out infinite;
}
.iris-ring-middle {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 60vmin;
    height: 60vmin;
    border: 2px solid rgba(255, 159, 28, 0.15);
    border-radius: 50%;
    animation: irisPulse 12s ease-in-out infinite reverse;
}
.iris-ring-inner {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 40vmin;
    height: 40vmin;
    border: 2px solid rgba(255, 191, 71, 0.2);
    border-radius: 50%;
    animation: irisPulse 8s ease-in-out infinite;
}
.iris-center {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 20vmin;
    height: 20vmin;
    background: radial-gradient(circle, rgba(255, 191, 71, 0.15) 0%, transparent 70%);
    border-radius: 50%;
    filter: blur(4vmin);
    animation: irisGlow 6s ease-in-out infinite;
}
@keyframes irisPulse {
    0%, 100% { 
        transform: translate(-50%, -50%) scale(1);
        opacity: 0.2;
    }
    50% { 
        transform: translate(-50%, -50%) scale(1.05);
        opacity: 0.4;
    }
}
@keyframes irisGlow {
    0%, 100% { 
        opacity: 0.3;
        filter: blur(4vmin);
    }
    50% { 
        opacity: 0.6;
        filter: blur(5vmin);
    }
}

/* === TYPOGRAPHY - CLEAN, NO GLOW === */
h1, h2, h3, h4 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600 !important;
    color: #FFBF47 !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 1.25rem !important;
    text-shadow: none !important;
}
h1 {
    font-size: 3.2rem !important;
    color: #FFBF47 !important;
    border-bottom: 2px solid #FF9F1C !important;
    padding-bottom: 0.75rem !important;
    margin-bottom: 2rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
h2 {
    font-size: 2.2rem !important;
    color: #FFBF47 !important;
    border-left: 4px solid #FF9F1C !important;
    padding-left: 1.2rem !important;
    font-weight: 600 !important;
}
h3 {
    font-size: 1.6rem !important;
    color: #E8D3B0 !important;
    font-weight: 500 !important;
}
h4 {
    font-size: 1.3rem !important;
    color: #B8A07E !important;
    font-weight: 500 !important;
}

p, li, .stMarkdown {
    font-size: 1.1rem !important;
    color: #CBB89E !important;
    line-height: 1.7 !important;
    font-weight: 400 !important;
}

/* === CARD DESIGNS - MINIMAL GLOW ON HOVER ONLY === */
.quantum-card, .bio-card {
    background: #14181C !important;
    border: 1px solid #2A2F35 !important;
    border-radius: 24px !important;
    padding: 1.8rem !important;
    box-shadow: 0 10px 30px -15px rgba(0, 0, 0, 0.8) !important;
    margin-bottom: 1.2rem !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
.quantum-card:hover, .bio-card:hover {
    border-color: #FFBF47 !important;
    box-shadow: 0 15px 35px -15px rgba(255, 191, 71, 0.2) !important;
    transform: translateY(-2px) !important;
    background: #1A1F24 !important;
}
.bio-card-glow {
    border: 1px solid #FFBF47 !important;
}
.bio-card-glow:hover {
    box-shadow: 0 15px 35px -15px rgba(255, 191, 71, 0.3) !important;
}
.bio-card-purple {
    border: 1px solid #B8A07E !important;
}
.bio-card-purple:hover {
    border-color: #E8D3B0 !important;
}
.bio-card-success {
    background: #1A241C !important;
    border: 1px solid #7A9F7A !important;
}
.bio-card-success:hover {
    border-color: #9FC79F !important;
    background: #1E2A1E !important;
}
.bio-card-danger {
    background: #241A1A !important;
    border: 1px solid #9F7A7A !important;
}
.bio-card-danger:hover {
    border-color: #CF9F9F !important;
    background: #2A1E1E !important;
}

/* === METRIC DISPLAYS - MINIMAL GLOW ON HOVER ONLY === */
.metric-neon, .stat-box, [data-testid="metric-container"] {
    background: #14181C !important;
    border: 1px solid #2A2F35 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    text-align: center !important;
    box-shadow: 0 5px 15px -10px rgba(0, 0, 0, 0.5) !important;
    transition: all 0.3s ease !important;
}
.metric-neon:hover, .stat-box:hover, [data-testid="metric-container"]:hover {
    border-color: #FFBF47 !important;
    box-shadow: 0 10px 25px -12px #FFBF47 !important;
    transform: translateY(-2px) !important;
    background: #1A1F24 !important;
}
.metric-neon-value, .stat-value, [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.2rem !important;
    font-weight: 500 !important;
    color: #FFBF47 !important;
    line-height: 1.2 !important;
    text-shadow: none !important;
}
.metric-neon-label, .stat-label, [data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    color: #B8A07E !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    font-weight: 400 !important;
    margin-top: 0.5rem !important;
}

/* === BADGES - NO GLOW, CLEAN COLORS === */
.badge, .crypto-chip {
    display: inline-block !important;
    padding: 0.4rem 1rem !important;
    border-radius: 40px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    border: 1px solid !important;
    transition: all 0.3s ease !important;
    background: transparent !important;
}
.badge:hover, .crypto-chip:hover {
    transform: translateY(-2px) !important;
}
.badge-success {
    border-color: #7A9F7A !important;
    color: #9FC79F !important;
}
.badge-success:hover {
    background: rgba(122, 159, 122, 0.15) !important;
}
.badge-danger {
    border-color: #9F7A7A !important;
    color: #CF9F9F !important;
}
.badge-danger:hover {
    background: rgba(159, 122, 122, 0.15) !important;
}
.badge-warn {
    border-color: #B8A07E !important;
    color: #E8D3B0 !important;
}
.badge-warn:hover {
    background: rgba(184, 160, 126, 0.15) !important;
}
.badge-info {
    border-color: #7A9FA0 !important;
    color: #9FC7C8 !important;
}
.badge-info:hover {
    background: rgba(122, 159, 160, 0.15) !important;
}

/* === BUTTONS - CLEAN HOVER === */
.stButton > button {
    background: transparent !important;
    color: #FFBF47 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600 !important;
    border: 2px solid #FFBF47 !important;
    border-radius: 40px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
}


/* Secondary button style */
.stButton > button[kind="secondary"] {
    border-color: #B8A07E !important;
    color: #B8A07E !important;
}
/* === BUTTON FOCUS STYLE === */
.stButton > button:focus,
.stButton > button:focus-visible {
    outline: none !important;
    background: rgba(255,191,71,0.08) !important;
    border: 2px solid #FFBF47 !important;
    box-shadow: 0 0 0 4px rgba(255,191,71,0.25) !important;
}

/* === BUTTON ACTIVE (when clicked) === */
.stButton > button:active {
    transform: scale(0.97) !important;
    background: rgba(255,191,71,0.12) !important;
}

/* === SECONDARY BUTTON FOCUS === */
.stButton > button[kind="secondary"]:focus,
.stButton > button[kind="secondary"]:focus-visible {
    outline: none !important;
    border-color: #B8A07E !important;
    box-shadow: 0 0 0 4px rgba(184,160,126,0.25) !important;
}

/* === INPUT FIELDS - CLEAN === */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #0F1317 !important;
    border: 2px solid #2A2F35 !important;
    border-radius: 12px !important;
    color: #E8D3B0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    padding: 0.8rem 1.2rem !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.5) !important;
    transition: all 0.3s ease !important;
}
.stTextInput > div > div > input:hover,
.stTextArea > div > div > textarea:hover,
.stNumberInput > div > div > input:hover,
.stSelectbox > div > div:hover {
    border-color: #FFBF47 !important;
    background: #141A20 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #FFBF47 !important;
    background: #1A2128 !important;
}
.stTextInput label, .stTextArea label, .stNumberInput label {
    font-size: 1rem !important;
    font-weight: 500 !important;
    color: #B8A07E !important;
    margin-bottom: 0.4rem !important;
    letter-spacing: 0.05em !important;
}
.stTextInput:focus-within label, .stTextArea:focus-within label, .stNumberInput:focus-within label {
    color: #FFBF47 !important;
}

/* === SIDEBAR - CLEAN === */
[data-testid="stSidebar"] {
    background: #0F1317 !important;
    border-right: 2px solid #2A2F35 !important;
    padding: 2rem 1.2rem !important;
    box-shadow: 2px 0 20px rgba(0, 0, 0, 0.5) !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #B8A07E !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    padding: 0.7rem 1.2rem !important;
    border-radius: 40px !important;
    margin: 0.3rem 0 !important;
    transition: all 0.3s ease !important;
    border: 2px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    border-color: #FFBF47 !important;
    color: #FFBF47 !important;
    background: rgba(255, 191, 71, 0.08) !important;
    transform: translateX(5px) !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdown"] {
    font-size: 1.1rem !important;
}

/* === AUTHENTICATION RESULT CARDS - CLEAN, NO GLOW === */
.auth-success, .result-auth {
    background: #1A241C !important;
    border: 2px solid #7A9F7A !important;
    border-radius: 32px !important;
    padding: 2.5rem !important;
    text-align: center !important;
    margin: 2rem 0 !important;
    transition: all 0.3s ease !important;
}
.auth-success:hover, .result-auth:hover {
    border-color: #9FC79F !important;
    transform: translateY(-2px) !important;
    background: #1F2B1F !important;
}
.auth-failure, .result-fail {
    background: #241A1A !important;
    border: 2px solid #9F7A7A !important;
    border-radius: 32px !important;
    padding: 2.5rem !important;
    text-align: center !important;
    margin: 2rem 0 !important;
    transition: all 0.3s ease !important;
}
.auth-failure:hover, .result-fail:hover {
    border-color: #CF9F9F !important;
    transform: translateY(-2px) !important;
    background: #2B1F1F !important;
}
.result-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.5rem !important;
    font-weight: 600 !important;
    color: #FFBF47 !important;
    letter-spacing: 0.1em !important;
    text-shadow: none !important;
}

/* === PROGRESS & SIMILARITY BARS - CLEAN === */
.feat-row {
    display: flex !important;
    align-items: center !important;
    gap: 1.2rem !important;
    margin-bottom: 1rem !important;
    padding: 0.4rem !important;
    transition: all 0.3s ease !important;
}
.feat-row:hover {
    background: rgba(255, 191, 71, 0.05) !important;
    border-radius: 12px !important;
}
.feat-label {
    font-size: 1rem !important;
    color: #B8A07E !important;
    min-width: 100px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
}
.feat-track {
    flex: 1 !important;
    height: 6px !important;
    background: #1E2329 !important;
    border-radius: 3px !important;
    overflow: hidden !important;
    border: 1px solid #2A2F35 !important;
}
.feat-fill {
    height: 100% !important;
    background: linear-gradient(90deg, #FF9F1C, #FFBF47) !important;
    border-radius: 3px !important;
    transition: width 0.6s ease !important;
}
.feat-score {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #FFBF47 !important;
    min-width: 75px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* === TABLE STYLES - CLEAN === */
.bio-table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-size: 1rem !important;
}
.bio-table th {
    background: #0F1317 !important;
    color: #B8A07E !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    padding: 1.2rem 1rem !important;
    border-bottom: 2px solid #FFBF47 !important;
    font-size: 0.9rem !important;
}
.bio-table td {
    padding: 1rem !important;
    border-bottom: 1px solid #2A2F35 !important;
    color: #CBB89E !important;
    background: #14181C !important;
    transition: all 0.3s ease !important;
}
.bio-table tr:hover td {
    background: #1F262E !important;
    color: #E8D3B0 !important;
}

/* === EXPANDER - CLEAN === */
.streamlit-expanderHeader {
    background: #0F1317 !important;
    border: 2px solid #2A2F35 !important;
    border-radius: 40px !important;
    color: #B8A07E !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 500 !important;
    font-size: 1.1rem !important;
    padding: 0.8rem 1.4rem !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.1em !important;
}
.streamlit-expanderHeader:hover {
    border-color: #FFBF47 !important;
    color: #FFBF47 !important;
    background: #1A2128 !important;
}
.streamlit-expanderContent {
    background: #14181C !important;
    border: 2px solid #2A2F35 !important;
    border-top: none !important;
    border-radius: 0 0 24px 24px !important;
    padding: 2rem !important;
    margin-top: -5px !important;
}

/* === PROGRESS BAR - CLEAN === */
.stProgress > div > div {
    background: linear-gradient(90deg, #FF9F1C, #FFBF47) !important;
    height: 6px !important;
    border-radius: 3px !important;
}
.stProgress > div {
    background: #1E2329 !important;
    border-radius: 3px !important;
    height: 6px !important;
    border: 1px solid #2A2F35 !important;
}

/* === CODE BLOCKS - CLEAN === */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    background: #0F1317 !important;
    border: 2px solid #2A2F35 !important;
    border-radius: 8px !important;
    color: #FFBF47 !important;
    font-size: 0.9rem !important;
    padding: 0.2rem 0.5rem !important;
    transition: all 0.3s ease !important;
}
code:hover, pre:hover {
    border-color: #FFBF47 !important;
    background: #141C24 !important;
}
pre {
    padding: 1.2rem !important;
    color: #CBB89E !important;
}

/* === TERMINAL - CLEAN === */
.terminal {
    background: #0F1317 !important;
    border: 2px solid #2A2F35 !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    color: #7A9F7A !important;
    max-height: 300px !important;
    overflow-y: auto !important;
    transition: all 0.3s ease !important;
}
.terminal:hover {
    border-color: #7A9F7A !important;
    background: #141E24 !important;
}

/* === FILE UPLOADER - CLEAN === */
[data-testid="stFileUploader"] {
    background: #0F1317 !important;
    border: 2px dashed #2A2F35 !important;
    border-radius: 24px !important;
    padding: 2.5rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #FFBF47 !important;
    background: #151D25 !important;
    transform: translateY(-2px) !important;
}

/* === DIVIDERS === */
hr {
    border: none !important;
    height: 2px !important;
    background: linear-gradient(90deg, transparent, #FFBF47, #FF9F1C, #FFBF47, transparent) !important;
    margin: 2.5rem 0 !important;
    opacity: 0.3 !important;
}

/* === SCROLLBAR - CLEAN === */
::-webkit-scrollbar {
    width: 10px !important;
    height: 10px !important;
}
::-webkit-scrollbar-track {
    background: #0F1317 !important;
    border: 1px solid #2A2F35 !important;
}
::-webkit-scrollbar-thumb {
    background: #2A2F35 !important;
    border-radius: 5px !important;
    transition: all 0.3s ease !important;
}
::-webkit-scrollbar-thumb:hover {
    background: #FFBF47 !important;
}

/* === ALERT BOXES - CLEAN === */
.stAlert {
    border-radius: 16px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 1.2rem !important;
    border-left: 6px solid !important;
    transition: all 0.3s ease !important;
}
.stAlert:hover {
    transform: translateX(5px) !important;
}
.stAlert[data-baseweb="alert"] {
    background: #14181C !important;
}

/* === GLOW DOT - SUBTLE === */
.glow-dot {
    display: inline-block !important;
    width: 8px !important;
    height: 8px !important;
    border-radius: 50% !important;
    background: #FFBF47 !important;
    margin-right: 0.8rem !important;
}

/* === TABS - CLEAN === */
.stTabs [data-baseweb="tab-list"] {
    gap: 2.5rem !important;
    border-bottom: 2px solid #2A2F35 !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    color: #B8A07E !important;
    transition: all 0.3s ease !important;
    padding: 0.8rem 0 !important;
    letter-spacing: 0.1em !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #FFBF47 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #FFBF47 !important;
    border-bottom-color: #FFBF47 !important;
}

/* === FOCUS STATES === */
*:focus-visible {
    outline: 2px solid #FFBF47 !important;
    outline-offset: 3px !important;
}

/* === RESPONSIVE ADJUSTMENTS === */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem !important;
    }
    h1 {
        font-size: 2.5rem !important;
    }
    h2 {
        font-size: 1.8rem !important;
    }
}
</style>

<!-- BIOLUMINESCENT IRIS ANIMATION (SCAN LINE REMOVED) -->
<div class="iris-container">
    <div class="iris-ring-outer"></div>
    <div class="iris-ring-middle"></div>
    <div class="iris-ring-inner"></div>
    <div class="iris-center"></div>
</div>
""", unsafe_allow_html=True)
# ── Session state init ─────────────────────────────────────────────────────────
for key, val in {
    "page": "home",
    "admin_logged_in": False,
    "database": None,           # raw JSON data
    "encrypted_database": None, # {uid: {public_key, encrypted_features}}
    "user_private_keys": None,  # {uid: serialized_private_key}
    "auth_result": None,
    "db_user_count": 0,
    "tenseal_available": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Check TenSEAL ──────────────────────────────────────────────────────────────
try:
    import tenseal as ts
    st.session_state.tenseal_available = True
except ImportError:
    st.session_state.tenseal_available = False

# ══════════════════════════════════════════════════════════════════════════════
#  EXACT FUNCTIONS FROM NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════

def normalize_vector(v):
    """EXACT from notebook Cell 3 & Cell 8"""
    v = np.array(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm > 0 else v.tolist()


def cosine_similarity_encrypted_public(v1, v2):
    """EXACT from notebook Cell 6"""
    import tenseal as ts
    dot_product = v1.dot(v2)
    norm_sq_v1 = v1.dot(v1)
    norm_sq_v2 = v2.dot(v2)
    return dot_product, norm_sq_v1, norm_sq_v2


def build_encrypted_database(data, progress_callback=None):
    import tenseal as ts

    user_private_keys = {}
    encrypted_database = {}
    logs = []

    for idx, (user_id, user_data) in enumerate(data.items()):
        if progress_callback:
            progress_callback(idx / len(data), f"Processing User {user_id}...")

        try:
            # Generate encryption context — 
            user_context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            user_context.generate_galois_keys()
            user_context.global_scale = 2 ** 40

            # Store private key
            user_private_key = user_context.serialize(save_secret_key=True)
            user_private_keys[user_id] = user_private_key
            logs.append(f"  ✓ Private key stored for user {user_id}")

            # Create public context
            user_public_context = user_context.copy()
            user_public_context.make_context_public()

            # NORMALIZE EACH VECTOR BEFORE ENCRYPTION — EXACT from notebook
            user_encrypted_features = []
            for vec in user_data["features"]:
                normalized_vec = normalize_vector(vec)
                enc_vec = ts.ckks_vector(user_public_context, normalized_vec)
                user_encrypted_features.append(enc_vec.serialize())

            # Store in database
            user_public_key = user_public_context.serialize()
            encrypted_database[user_id] = {
                'public_key': user_public_key,
                'encrypted_features': user_encrypted_features
            }
            logs.append(f"  ✓ Added User {user_id}: {len(user_encrypted_features)} NORMALIZED features")

        except Exception as e:
            logs.append(f"  ✗ Error with User {user_id}: {e}")
            continue

    logs.append(f"✓ Enrollment completed. Stored private keys for {len(user_private_keys)} users")
    return encrypted_database, user_private_keys, logs


def authenticate_user(query_user_id, query_features_raw, encrypted_database, user_private_keys):
    """EXACT logic from notebook Cells 8–11"""
    import tenseal as ts

    # ── Cell 8: Normalize query features ──────────────────────────────────────
    normalized_queries = []
    for feat in query_features_raw:
        normalized = normalize_vector(feat)
        normalized_queries.append(normalized)

    # Retrieve target user's public context
    target_data = encrypted_database[query_user_id]
    target_public_context = ts.context_from(target_data['public_key'])

    # Encrypt NORMALIZED query vectors
    enc_queries = []
    for query_vec in normalized_queries:
        enc_query = ts.ckks_vector(target_public_context, query_vec)
        enc_queries.append(enc_query)

    # Retrieve private context
    user_private_context = ts.context_from(user_private_keys[query_user_id])

    # ── Cell 9: Server-side comparison ────────────────────────────────────────
    target_enc_features = target_data['encrypted_features']
    encrypted_results_all = []

    for template_idx, enc_feat_serialized in enumerate(target_enc_features):
        try:
            enc_feat = ts.ckks_vector_from(target_public_context, enc_feat_serialized)
            template_results = []

            for enc_query in enc_queries:
                dot_product, norm_sq_v1, norm_sq_v2 = cosine_similarity_encrypted_public(enc_query, enc_feat)
                template_results.append({
                    'dot_product': dot_product.serialize(),
                    'norm_sq_v1': norm_sq_v1.serialize(),
                    'norm_sq_v2': norm_sq_v2.serialize()
                })

            encrypted_results_all.append(template_results)
        except Exception as e:
            encrypted_results_all.append([])

    # ── Cell 10: Client-side decryption with XAI data collection ──────────────
    decrypted_average_scores = []
    feature_similarities_all = []
    threshold_value = 0.999
    templates_with_high_feature = []

    for template_idx, template_results in enumerate(encrypted_results_all):
        if not template_results:
            decrypted_average_scores.append(0.0)
            feature_similarities_all.append([])
            continue

        try:
            feature_similarities = []

            for result in template_results:
                dot_product_enc   = ts.ckks_vector_from(user_private_context, result['dot_product'])
                norm_sq_v1_enc    = ts.ckks_vector_from(user_private_context, result['norm_sq_v1'])
                norm_sq_v2_enc    = ts.ckks_vector_from(user_private_context, result['norm_sq_v2'])

                dot_product     = dot_product_enc.decrypt()[0]
                norm_sq_v1_val  = norm_sq_v1_enc.decrypt()[0]
                norm_sq_v2_val  = norm_sq_v2_enc.decrypt()[0]

                norm_sq_v1_val = max(0, norm_sq_v1_val)
                norm_sq_v2_val = max(0, norm_sq_v2_val)
                norm_v1 = np.sqrt(norm_sq_v1_val) if norm_sq_v1_val > 0 else 0
                norm_v2 = np.sqrt(norm_sq_v2_val) if norm_sq_v2_val > 0 else 0

                if norm_v1 == 0 or norm_v2 == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm_v1 * norm_v2)

                similarity = max(-1.0, min(1.0, similarity))
                feature_similarities.append(similarity)

            feature_similarities_all.append(feature_similarities)
            average_similarity = sum(feature_similarities) / len(feature_similarities)
            decrypted_average_scores.append(average_similarity)

            high_features = [sim for sim in feature_similarities if sim >= threshold_value]
            if high_features:
                templates_with_high_feature.append({
                    'template': template_idx + 1,
                    'high_features': high_features,
                    'average': average_similarity
                })

        except Exception as e:
            decrypted_average_scores.append(0.0)
            feature_similarities_all.append([])

    # ── Cell 11: XAI analysis ─────────────────────────────────────────────────
    threshold_value_xai = 0.999
    min_templates_required = 2
    average_threshold = 0.5

    templates_with_high_features = []
    high_feature_counts = []

    for template_idx, features in enumerate(feature_similarities_all):
        if features:
            high_features = [sim for sim in features if sim >= threshold_value_xai]
            if high_features:
                templates_with_high_features.append({
                    'template': template_idx + 1,
                    'high_features': high_features,
                    'count': len(high_features),
                    'max_feature': max(features),
                    'average': decrypted_average_scores[template_idx] if template_idx < len(decrypted_average_scores) else 0
                })
                high_feature_counts.append(len(high_features))

    num_templates_with_high_features = len(templates_with_high_features)
    auth_success = num_templates_with_high_features >= min_templates_required

    # ── Enhanced AuthenticationXI class logic (Cell 12 final) ─────────────────
    num_features = len(feature_similarities_all[0]) if feature_similarities_all and feature_similarities_all[0] else 0
    min_templates_xi = num_features

    templates_meeting_rule = []
    for idx, features in enumerate(feature_similarities_all):
        if features:
            high_f = [float(f) for f in features if f >= 0.999]
            if high_f:
                templates_meeting_rule.append({
                    'template': idx + 1,
                    'high_features': high_f,
                    'count': len(high_f),
                    'max_feature': float(max(features)),
                    'avg_score': float(decrypted_average_scores[idx]) if idx < len(decrypted_average_scores) else 0
                })

    rule_based_success = len(templates_meeting_rule) >= min_templates_xi

    best_template_idx = int(np.argmax(decrypted_average_scores)) if decrypted_average_scores else 0
    best_score = decrypted_average_scores[best_template_idx] if decrypted_average_scores else 0.0
    is_authenticated = best_score >= 0.70

    # Near misses
    near_misses = []
    for idx, features in enumerate(feature_similarities_all):
        if features:
            near = [float(f) for f in features if 0.94 <= f < 0.999]
            if near:
                near_misses.append({
                    'template': idx + 1,
                    'near_features': near,
                    'best_feature': float(max(features)),
                    'gap': float(0.999 - max(features))
                })

    def get_confidence(s):
        if s >= 0.99: return "VERY HIGH"
        if s >= 0.90: return "HIGH"
        if s >= 0.80: return "MODERATE"
        if s >= 0.60: return "LOW"
        return "VERY LOW"

    def get_rule_confidence(n, req):
        ratio = n / req if req > 0 else 0
        if ratio >= 1.5: return "VERY HIGH"
        if ratio >= 1.0: return "HIGH"
        if ratio >= 0.75: return "MODERATE"
        if ratio >= 0.5: return "LOW"
        return "VERY LOW"

    def get_action():
        if rule_based_success and is_authenticated:
            return {"action": "GRANT FULL ACCESS", "reason": f"Both methods confirm identity ({len(templates_meeting_rule)}/{min_templates_xi} templates)", "severity": "LOW"}
        if rule_based_success and not is_authenticated:
            return {"action": "GRANT FULL ACCESS", "reason": "High-confidence features but avg score below threshold", "severity": "MEDIUM", "suggestion": "Consider lowering average threshold"}
        if not rule_based_success and is_authenticated:
            return {"action": "DENY ACCESS", "reason": f"Avg meets threshold but lacks high-confidence features ({len(templates_meeting_rule)}/{min_templates_xi})", "severity": "MEDIUM", "suggestion": "Consider re-enrollment"}
        if best_score >= 0.65:
            return {"action": "REQUEST MORE FEATURES", "reason": f"Close but insufficient ({len(templates_meeting_rule)}/{min_templates_xi} templates)", "severity": "MEDIUM"}
        return {"action": "DENY ACCESS", "reason": f"Failed both methods. Need {max(0,min_templates_xi-len(templates_meeting_rule))} more templates", "severity": "HIGH", "suggestion": "Security alert"}

    xai_struct = {
        "query_user_id": query_user_id,
        "configuration": {
            "num_features": num_features,
            "feature_threshold": 0.999,
            "min_templates_required": min_templates_xi,
            "average_threshold": 0.70
        },
        "authentication_rules": {
            "rule_based": {
                "success": rule_based_success,
                "templates_meeting_rule": len(templates_meeting_rule),
                "templates_detail": templates_meeting_rule
            },
            "traditional": {
                "best_template": best_template_idx + 1,
                "best_score": float(best_score),
                "authenticated": is_authenticated,
                "margin": float(best_score - 0.70),
                "templates_above_threshold": int(sum(1 for s in decrypted_average_scores if s >= 0.70))
            }
        },
        "summary_statistics": {
            "templates_analyzed": len(decrypted_average_scores),
            "avg_score_all_templates": float(np.mean(decrypted_average_scores)) if decrypted_average_scores else 0,
            "score_variance": float(np.var(decrypted_average_scores)) if decrypted_average_scores else 0,
            "min_score": float(min(decrypted_average_scores)) if decrypted_average_scores else 0,
            "max_score": float(max(decrypted_average_scores)) if decrypted_average_scores else 0
        },
        "confidence_metrics": {
            "rule_based_confidence": get_rule_confidence(len(templates_meeting_rule), min_templates_xi),
            "traditional_confidence": get_confidence(best_score),
            "methods_agree": rule_based_success == is_authenticated
        },
        "recommended_action": get_action(),
        "security_indicators": {
            "has_near_misses": len(near_misses) > 0,
            "near_misses_detail": near_misses,
            "high_confidence_templates": len(templates_meeting_rule),
            "total_high_features": sum(t["count"] for t in templates_meeting_rule),
            "templates_shortfall": max(0, min_templates_xi - len(templates_meeting_rule)) if not rule_based_success else 0
        },
        "feature_analysis": {}
    }

    if feature_similarities_all and len(feature_similarities_all) > best_template_idx:
        feats = feature_similarities_all[best_template_idx]
        if feats:
            xai_struct["feature_analysis"] = {
                "best_template_features": [float(f) for f in feats],
                "num_features": len(feats),
                "high_confidence_features": sum(1 for f in feats if f >= 0.999),
                "strong_features": sum(1 for f in feats if f >= 0.7),
                "moderate_features": sum(1 for f in feats if 0.5 <= f < 0.7),
                "weak_features": sum(1 for f in feats if f < 0.5)
            }

    return {
        "authenticated": rule_based_success,
        "auth_success_xai": auth_success,
        "best_score": best_score,
        "template_scores": decrypted_average_scores,
        "feature_similarities_all": feature_similarities_all,
        "templates_with_high_features": templates_with_high_features,
        "num_templates_with_high_features": num_templates_with_high_features,
        "min_templates_required": min_templates_required,
        "xai_struct": xai_struct
    }


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def generate_comprehensive_xai_pdf(xai_struct, decrypted_average_scores, feature_similarities_all, query_user_id, min_templates_required):
    """
    EXACT function from notebook Cell 12 — generate_comprehensive_xai_pdf()
    Generates full PDF with charts using reportlab + matplotlib.
    Returns bytes buffer ready for st.download_button.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER

    # Convert all numpy types to native Python types first
    xai_struct = json.loads(json.dumps(xai_struct, cls=NumpyEncoder))

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()

    # Custom styles — EXACT from notebook
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.darkblue,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        textColor=colors.darkblue,
        fontName='Helvetica-Bold',
        borderWidth=1,
        borderColor=colors.darkblue,
        borderPadding=5,
        backColor=colors.lightblue
    )

    story = []

    # ── TITLE PAGE ── EXACT from notebook
    story.append(Paragraph("PRIVACY-PRESERVING BIOMETRIC AUTHENTICATION", title_style))
    story.append(Paragraph("<b>Interpretable Verification Analysis Report</b>", title_style))
    story.append(Spacer(1, 0.3*inch))

    metadata = [
        ["Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Query User ID:", str(query_user_id)],
        ["Number of Features:", str(xai_struct['configuration']['num_features'])],
        ["Rule:", f">=1 feature >=0.99 in >={min_templates_required} templates"],
        ["System:", "Fully Homomorphic Encryption (FHE) Authentication"],
        
    ]
    meta_table = Table(metadata, colWidths=[2*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 10)
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3*inch))

    # ── SECTION 1: QUICK SUMMARY — EXACT from notebook
    story.append(Paragraph("1. QUICK ANALYSIS", heading_style))

    best_score    = xai_struct['authentication_rules']['traditional']['best_score']
    threshold     = xai_struct['configuration']['average_threshold']
    rule_success  = xai_struct['authentication_rules']['rule_based']['success']
    trad_success  = xai_struct['authentication_rules']['traditional']['authenticated']
    templates_meeting = xai_struct['authentication_rules']['rule_based']['templates_detail']

    quick_summary = f"""
    <b>User:</b> {query_user_id}<br/>
    <b>Best Score (Average):</b> {best_score:.4f}<br/>
    <b>Rule-Based Result:</b> {'<font color="green">AUTHENTICATED</font>' if rule_success else '<font color="red">REJECTED</font>'}<br/>
    <b>Average Threshold:</b> {threshold}<br/>
    <b>Average Result:</b> {'<font color="green">AUTHENTICATED</font>' if trad_success else '<font color="red">REJECTED</font>'}<br/>
    <b>Templates with High Features (>=0.99):</b> {len(templates_meeting)}/{len(decrypted_average_scores)}<br/>
    <b>Required Templates:</b> {min_templates_required}<br/>
    <b>Methods Agree:</b> {'Yes' if xai_struct['confidence_metrics']['methods_agree'] else 'No'}<br/>
    <b>Action:</b> {xai_struct['recommended_action']['action']}<br/>
    <b>Reason:</b> {xai_struct['recommended_action']['reason']}
    """
    story.append(Paragraph(quick_summary, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # ── SECTION 2: DETAILED TEMPLATE ANALYSIS — EXACT from notebook
    story.append(Paragraph("2. DETAILED TEMPLATE ANALYSIS", heading_style))

    template_data = [["Template", "Avg Score", "Has High Feature", "High Features", "Max Feature"]]
    for idx, (score, features) in enumerate(zip(decrypted_average_scores, feature_similarities_all)):
        if features:
            high_features = [f for f in features if f >= 0.999]
            has_high = "YES" if high_features else "NO"
            high_count = len(high_features)
            max_feat = max(features)
            template_data.append([str(idx+1), f"{score:.4f}", has_high, str(high_count), f"{max_feat:.4f}"])

    template_table = Table(template_data)
    template_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9)
    ]))
    story.append(template_table)
    story.append(Spacer(1, 0.2*inch))

    # ── SECTION 3: RULE-BASED ANALYSIS — EXACT from notebook
    story.append(Paragraph("3. RULE-BASED AUTHENTICATION ANALYSIS", heading_style))

    rule_text = f"""
    <b>Rule:</b> At least one feature >= 0.99 in at least {min_templates_required} templates<br/>
    <b>Templates meeting rule:</b> {len(templates_meeting)}/{len(decrypted_average_scores)}<br/>
    <b>Result:</b> {'<font color="green">SUCCESS</font>' if rule_success else '<font color="red">FAILURE</font>'}<br/><br/>
    <b>Templates with high-confidence features:</b><br/>
    """
    for t in templates_meeting:
        rule_text += f"- Template {t['template']}: {t['count']} high feature(s) - Max: {t['max_feature']:.4f}<br/>"
    if not templates_meeting:
        rule_text += "- No templates have features >=0.99<br/>"

    near_misses = xai_struct['security_indicators']['near_misses_detail']
    if near_misses:
        rule_text += "<br/><b>Near misses detected:</b><br/>"
        for nm in near_misses:
            rule_text += f"- Template {nm['template']}: best feature = {nm['best_feature']:.4f} (needs {nm['gap']:.4f} more)<br/>"

    story.append(Paragraph(rule_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # ── SECTION 4: TRADITIONAL ANALYSIS — EXACT from notebook
    story.append(Paragraph("4.AVERAGE THRESHOLD-BASED ANALYSIS", heading_style))

    traditional_text = f"""
    <b>Best Template:</b> {xai_struct['authentication_rules']['traditional']['best_template']}<br/>
    <b>Best Score:</b> {best_score:.4f}<br/>
    <b>Threshold:</b> {threshold}<br/>
    <b>Margin:</b> {xai_struct['authentication_rules']['traditional']['margin']:+.4f}<br/>
    <b>Result:</b> {'<font color="green">AUTHENTICATED</font>' if trad_success else '<font color="red">REJECTED</font>'}<br/>
    <b>Templates above threshold:</b> {xai_struct['authentication_rules']['traditional']['templates_above_threshold']}/{len(decrypted_average_scores)}<br/>
    """
    story.append(Paragraph(traditional_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # ── SECTION 5: FEATURE ANALYSIS — EXACT from notebook
    if xai_struct['feature_analysis']:
        story.append(Paragraph("5. FEATURE-LEVEL ANALYSIS", heading_style))

        best_template_idx = xai_struct['authentication_rules']['traditional']['best_template']
        features = xai_struct['feature_analysis']['best_template_features']

        feature_table_data = [["Feature", "Score", "Status"]]
        for i, score in enumerate(features):
            if score >= 0.99:   status = "HIGH CONFIDENCE"
            elif score >= 0.7:  status = "STRONG"
            elif score >= 0.5:  status = "MODERATE"
            else:               status = "WEAK"
            feature_table_data.append([f"Feature {i+1}", f"{score:.4f}", status])

        feature_table = Table(feature_table_data)
        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(feature_table)

        summary_text = f"""
        <br/><b>Feature Summary:</b><br/>
        - High confidence features (>=0.99): {xai_struct['feature_analysis']['high_confidence_features']}<br/>
        - Strong features (>=0.7): {xai_struct['feature_analysis']['strong_features']}<br/>
        - Moderate features (0.5-0.7): {xai_struct['feature_analysis']['moderate_features']}<br/>
        - Weak features (<0.5): {xai_struct['feature_analysis']['weak_features']}<br/>
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

    # ── SECTION 6: CONFIDENCE & RECOMMENDATIONS — EXACT from notebook
    story.append(Paragraph("6. CONFIDENCE ASSESSMENT & RECOMMENDATIONS", heading_style))

    confidence_text = f"""
    <b>Average-Based Confidence:</b> {xai_struct['confidence_metrics']['traditional_confidence']}<br/>
    <b>Rule-Based Confidence:</b> {xai_struct['confidence_metrics']['rule_based_confidence']}<br/>
    <b>Methods Agree:</b> {'Yes' if xai_struct['confidence_metrics']['methods_agree'] else 'No'}<br/><br/>
    <b>Recommended Action:</b> {xai_struct['recommended_action']['action']}<br/>
    <b>Reason:</b> {xai_struct['recommended_action']['reason']}<br/>
    """
    if 'suggestion' in xai_struct['recommended_action']:
        confidence_text += f"<b>Suggestion:</b> {xai_struct['recommended_action']['suggestion']}<br/>"
    story.append(Paragraph(confidence_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # ── SECTION 7: SECURITY INDICATORS — EXACT from notebook
    story.append(Paragraph("7. SECURITY INDICATORS", heading_style))

    security_text = f"""
    <b>High Confidence Templates:</b> {xai_struct['security_indicators']['high_confidence_templates']}<br/>
    <b>Total High Features Detected:</b> {xai_struct['security_indicators']['total_high_features']}<br/>
    <b>Near Misses Detected:</b> {'Yes' if xai_struct['security_indicators']['has_near_misses'] else 'No'}<br/>
    <b>Templates Shortfall:</b> {xai_struct['security_indicators']['templates_shortfall']}<br/>
    """
    story.append(Paragraph(security_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # ── SECTION 8: VISUALIZATIONS — EXACT 6 charts from notebook
    story.append(PageBreak())
    story.append(Paragraph("8. VISUAL ANALYSIS", heading_style))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.patch.set_facecolor('white')

    template_indices = range(1, len(decrypted_average_scores) + 1)

    # Chart 1: Template scores — EXACT from notebook
    colors_bar = ['green' if any(f >= 0.999 for f in (feature_similarities_all[i-1] or [])) else 'skyblue'
                  for i in template_indices]
    axes[0, 0].bar(template_indices, decrypted_average_scores, color=colors_bar, alpha=0.7)
    axes[0, 0].axhline(y=0.70, color='red', linestyle='--', linewidth=2, label='Threshold (0.70)')
    axes[0, 0].set_title('Template Scores (Green = has feature >=0.99)')
    axes[0, 0].set_xlabel('Template ID')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Chart 2: Templates meeting rule — EXACT from notebook
    templates_count = [0] * len(decrypted_average_scores)
    for t in templates_meeting:
        templates_count[t['template']-1] = t['count']
    axes[0, 1].bar(template_indices, templates_count, color='orange', alpha=0.7)
    axes[0, 1].axhline(y=min_templates_required, color='red', linestyle='--', linewidth=2,
                        label=f'Required ({min_templates_required})')
    axes[0, 1].set_title('Templates Meeting Rule (>=0.99 features)')
    axes[0, 1].set_xlabel('Template ID')
    axes[0, 1].set_ylabel('Number of High Features')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Chart 3: Confidence comparison — EXACT from notebook
    conf_map = {'VERY HIGH': 0.95, 'HIGH': 0.85, 'MODERATE': 0.70, 'LOW': 0.50, 'VERY LOW': 0.20,
                'VERY_HIGH': 0.95, 'VERY_LOW': 0.20}
    trad_conf = conf_map.get(xai_struct['confidence_metrics']['traditional_confidence'], 0.5)
    rule_conf  = conf_map.get(xai_struct['confidence_metrics']['rule_based_confidence'],  0.5)
    axes[0, 2].bar([1, 2], [trad_conf, rule_conf], color=['#4CAF50', '#FF9800'], alpha=0.8)
    axes[0, 2].set_xticks([1, 2])
    axes[0, 2].set_xticklabels(['Average', 'Rule-Based'])
    axes[0, 2].set_title('Confidence Comparison')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([trad_conf, rule_conf]):
        axes[0, 2].text(i+1, v+0.02, f'{v*100:.0f}%', ha='center', fontweight='bold')

    # Chart 4: Feature distribution best template — EXACT from notebook
    if xai_struct['feature_analysis'].get('best_template_features'):
        feats = xai_struct['feature_analysis']['best_template_features']
        feat_colors = ['gold' if f>=0.99 else 'green' if f>=0.7 else 'orange' if f>=0.5 else 'red' for f in feats]
        axes[1, 0].bar(range(1, len(feats)+1), feats, color=feat_colors, alpha=0.8)
        axes[1, 0].axhline(y=0.99, color='gold', linestyle='--', linewidth=2, label='High (>=0.99)')
        axes[1, 0].axhline(y=0.7,  color='green', linestyle='--', label='Strong (>=0.7)')
        axes[1, 0].axhline(y=0.5,  color='orange', linestyle=':', label='Moderate (>=0.5)')
        axes[1, 0].set_title(f'Features (Best Template {xai_struct["authentication_rules"]["traditional"]["best_template"]})')
        axes[1, 0].set_xlabel('Feature')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No feature data', ha='center', va='center')
        axes[1, 0].set_title('Feature Analysis')

    # Chart 5: Methods agreement pie — EXACT from notebook
    methods_agree = xai_struct['confidence_metrics']['methods_agree']
    axes[1, 1].pie(
        [1 if methods_agree else 0, 0 if methods_agree else 1],
        labels=['Agree' if methods_agree else 'Disagree', ''],
        colors=['#4CAF50' if methods_agree else '#FF9800', 'white'],
        autopct='%1.1f%%' if methods_agree else None,
        startangle=90
    )
    axes[1, 1].set_title('Methods Agreement')

    # Chart 6: Near misses — EXACT from notebook
    if near_misses:
        axes[1, 2].bar([nm['template'] for nm in near_misses],
                       [nm['gap'] for nm in near_misses], color='orange', alpha=0.8)
        axes[1, 2].set_title('Near Misses (Gap to 0.99)')
        axes[1, 2].set_xlabel('Template')
        axes[1, 2].set_ylabel('Gap to Threshold')
        axes[1, 2].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 2].text(0.5, 0.5, 'No near misses detected', ha='center', va='center',
                        transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Near Misses Analysis')

    plt.suptitle(f'Analysis — User {query_user_id} | Rule: Need {min_templates_required} templates with >=0.99',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    chart_buffer = io.BytesIO()
    plt.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight')
    chart_buffer.seek(0)
    chart_img = Image(chart_buffer, width=7*inch, height=5*inch)
    story.append(chart_img)
    plt.close()

    # ── SECTION 9: STRUCTURED JSON DATA — EXACT from notebook
    story.append(PageBreak())
    story.append(Paragraph("9. STRUCTURED DATA (JSON)", heading_style))

    json_text = json.dumps(xai_struct, indent=2, cls=NumpyEncoder)
    for i, line in enumerate(json_text.split('\n')):
        if i < 60:
            story.append(Paragraph(f"<font name='Courier' size=7>{line}</font>", styles['Code']))
        elif i == 60:
            story.append(Paragraph("<font name='Courier' size=7>...</font>", styles['Code']))
            break

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def feat_color(score):
    if score >= 0.999: return "#FFD700"
    if score >= 0.7:   return "#00E676"
    if score >= 0.5:   return "#FFB800"
    return "#FF4444"

def feat_label(score):
    if score >= 0.999: return "HIGH ⭐"
    if score >= 0.7:   return "STRONG"
    if score >= 0.5:   return "MODERATE"
    return "WEAK"

def render_feature_bars(features, title="Feature Analysis"):
    st.markdown(f"**{title}**")
    for i, score in enumerate(features):
        pct = max(0, min(100, score * 100))
        color = feat_color(score)
        label = feat_label(score)
        st.markdown(f"""
        <div class="feat-row">
            <span class="feat-label">Feature {i+1}</span>
            <div class="feat-track">
                <div class="feat-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <span class="feat-score" style="color:{color};">{score:.4f}</span>
            <span style="font-size:10px;color:{color};min-width:70px;">{label}</span>
        </div>
        """, unsafe_allow_html=True)

def render_template_bars(template_scores, feature_similarities_all):
    st.markdown("**Template Scores**")
    for i, score in enumerate(template_scores):
        has_high = any(f >= 0.999 for f in (feature_similarities_all[i] or []))
        pct = max(0, min(100, score * 100))
        color = "#00E676" if has_high else "#7B61FF"
        mark = " ✓" if has_high else ""
        st.markdown(f"""
        <div class="feat-row">
            <span class="feat-label">T{i+1}{mark}</span>
            <div class="feat-track">
                <div class="feat-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <span class="feat-score" style="color:{color};">{score:.4f}</span>
        </div>
        """, unsafe_allow_html=True)

def render_xai_full(result):
    """Render full XAI analysis panel"""
    xai = result["xai_struct"]
    rb  = xai["authentication_rules"]["rule_based"]
    trad = xai["authentication_rules"]["traditional"]
    conf = xai["confidence_metrics"]
    action = xai["recommended_action"]
    sec = xai["security_indicators"]
    cfg = xai["configuration"]
    stats = xai["summary_statistics"]

    # ── Stats row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Score", f"{trad['best_score']:.4f}")
    c2.metric("Templates Meeting Rule", f"{rb['templates_meeting_rule']}/{cfg['min_templates_required']}")
    c3.metric("Rule Confidence", conf["rule_based_confidence"])
    c4.metric("Trad. Confidence", conf["traditional_confidence"])

    st.markdown("---")

    # ── Template + Feature bars ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        render_template_bars(result["template_scores"], result["feature_similarities_all"])
    with col_b:
        if xai["feature_analysis"].get("best_template_features"):
            render_feature_bars(xai["feature_analysis"]["best_template_features"],
                                f"Feature Analysis (Best Template {trad['best_template']})")

    st.markdown("---")

    # ── Rule-based detail ──────────────────────────────────────────────────────
    st.markdown("**📋 Rule-Based Authentication Detail**")
    st.markdown(f"""
    <div class="bio-card">
        <p style="font-size:12px;color:#5A7A9A;margin-bottom:12px;">
        Rule: At least 1 feature ≥ {cfg['feature_threshold']} in ≥ {cfg['min_templates_required']} templates
        </p>
    """, unsafe_allow_html=True)

    if rb["templates_detail"]:
        for t in rb["templates_detail"]:
            badge = '<span class="badge badge-success">MEETS RULE ✓</span>'
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:10px 0;border-bottom:1px solid #1A2D4A;font-size:12px;">
                <span style="color:#E2EAF4;font-weight:600;">Template {t['template']}</span>
                {badge}
                <span style="color:#5A7A9A;">High features: {t['count']} | Max: <strong style="color:#FFD700;">{t['max_feature']:.4f}</strong></span>
                <span style="color:#5A7A9A;">Avg: {t['avg_score']:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#FF4444;font-size:12px;">No templates have features ≥ 0.99</p>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Traditional analysis ───────────────────────────────────────────────────
    st.markdown("**📈 Traditional Threshold Analysis**")
    cols = st.columns(4)
    cols[0].markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00E5FF;font-size:20px;">{trad["best_template"]}</div><div class="stat-label">Best Template</div></div>', unsafe_allow_html=True)
    cols[1].markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00E5FF;font-size:20px;">{trad["best_score"]:.4f}</div><div class="stat-label">Best Score</div></div>', unsafe_allow_html=True)
    cols[2].markdown(f'<div class="stat-box"><div class="stat-value" style="color:{"#00E676" if trad["margin"]>=0 else "#FF4444"};font-size:20px;">{trad["margin"]:+.4f}</div><div class="stat-label">Margin</div></div>', unsafe_allow_html=True)
    cols[3].markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00E5FF;font-size:20px;">{trad["templates_above_threshold"]}</div><div class="stat-label">Templates ≥ 0.70</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Method comparison ──────────────────────────────────────────────────────
    st.markdown("**⚖ Method Comparison**")
    mc1, mc2, mc3 = st.columns(3)
    rb_ok = rb["success"]
    tr_ok = trad["authenticated"]
    agree = conf["methods_agree"]

    mc1.markdown(f"""<div class="stat-box">
        <div class="stat-value" style="color:{'#00E676' if rb_ok else '#FF4444'};font-size:18px;">
        {'✅ PASS' if rb_ok else '❌ FAIL'}</div>
        <div class="stat-label">Rule-Based</div></div>""", unsafe_allow_html=True)
    mc2.markdown(f"""<div class="stat-box">
        <div class="stat-value" style="color:{'#00E676' if tr_ok else '#FF4444'};font-size:18px;">
        {'✅ PASS' if tr_ok else '❌ FAIL'}</div>
        <div class="stat-label">Traditional</div></div>""", unsafe_allow_html=True)
    mc3.markdown(f"""<div class="stat-box">
        <div class="stat-value" style="color:{'#00E676' if agree else '#FFB800'};font-size:18px;">
        {'✅ AGREE' if agree else '⚠ DISAGREE'}</div>
        <div class="stat-label">Methods</div></div>""", unsafe_allow_html=True)

    if not agree:
        if rb_ok and not tr_ok:
            st.info("⚠ Rule-based says YES (high-confidence features) but traditional says NO (avg below 0.70). Consider lowering average threshold.")
        elif not rb_ok and tr_ok:
            st.warning("⚠ Traditional says YES (avg above 0.70) but rule-based says NO (insufficient high-confidence features).")

    st.markdown("---")

    # ── Recommended action ─────────────────────────────────────────────────────
    sev_color = {"LOW": "#00E676", "MEDIUM": "#FFB800", "HIGH": "#FF4444"}.get(action.get("severity",""), "#5A7A9A")
    st.markdown(f"""
    <div class="bio-card" style="border-color:rgba({
        '0,230,118' if action.get('severity')=='LOW' else
        '255,184,0' if action.get('severity')=='MEDIUM' else
        '255,68,68'
    },0.3);">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
            <span class="badge badge-{'success' if action.get('severity')=='LOW' else 'warn' if action.get('severity')=='MEDIUM' else 'danger'}">
                {action.get('action','N/A')}
            </span>
        </div>
        <p style="font-size:12px;color:#E2EAF4;">{action.get('reason','')}</p>
        {"<p style='font-size:11px;color:#5A7A9A;margin-top:8px;'>💡 " + action.get('suggestion','') + "</p>" if action.get('suggestion') else ""}
    </div>
    """, unsafe_allow_html=True)

    # ── Near misses ────────────────────────────────────────────────────────────
    if sec["has_near_misses"]:
        st.markdown("**⚠ Near Misses Detected**")
        for nm in sec["near_misses_detail"]:
            st.markdown(f"""
            <div style="font-size:12px;padding:8px 0;border-bottom:1px solid #1A2D4A;
                        display:flex;justify-content:space-between;">
                <span style="color:#5A7A9A;">Template {nm['template']}</span>
                <span>Best: <strong style="color:#FFB800;">{nm['best_feature']:.4f}</strong></span>
                <span>Gap: <strong style="color:#FF4444;">+{nm['gap']:.4f}</strong> needed</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Summary statistics ─────────────────────────────────────────────────────
    with st.expander("📊 Summary Statistics"):
        st.json({
            "templates_analyzed": stats["templates_analyzed"],
            "avg_score": round(stats["avg_score_all_templates"], 4),
            "score_variance": round(stats["score_variance"], 6),
            "min_score": round(stats["min_score"], 4),
            "max_score": round(stats["max_score"], 4)
        })

    with st.expander("🗂 Full XAI JSON"):
        st.json(json.loads(json.dumps(xai, cls=NumpyEncoder)))


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown("""
    <div style="text-align:center;padding:60px 20px 40px;">
        <div style="display:inline-flex;align-items:center;gap:8px;padding:6px 18px;
                    background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.2);
                    border-radius:999px;margin-bottom:28px;">
            <span class="glow-dot"></span>
            <span style="font-size:11px;color:#00E5FF;letter-spacing:0.15em;">SYSTEM ONLINE</span>
        </div>
        <h1 style="margin-bottom:16px;">Privacy-Preserving<br/>Biometric Authentication</h1>
        <p style="color:#5A7A9A;font-size:14px;max-width:520px;margin:0 auto 48px;line-height:1.8;">
            Fully Homomorphic Encryption (FHE) + TenSEAL CKKS based biometric verification
            with Decision Analysis Report. Zero plaintext exposure.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="bio-card bio-card-glow" style="text-align:center;cursor:pointer;">
            <div style="font-size:40px;margin-bottom:16px;">🔐</div>
            <h3>Admin Portal</h3>
            <p style="font-size:12px;color:#5A7A9A;line-height:1.7;margin-bottom:20px;">
                Upload & manage the biometric database.<br/>
                Build encrypted user templates with TenSEAL CKKS.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Login as Admin →", key="btn_admin", use_container_width=True):
            st.session_state.page = "admin_login"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="bio-card bio-card-purple" style="text-align:center;cursor:pointer;">
            <div style="font-size:40px;margin-bottom:16px;">👤</div>
            <h3 style="color:#7B61FF;">User Portal</h3>
            <p style="font-size:12px;color:#5A7A9A;line-height:1.7;margin-bottom:20px;">
                Upload your feature vector for encrypted authentication.<br/>
                View Detailed Decision Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter as User →", key="btn_user", use_container_width=True):
            st.session_state.page = "user"
            st.rerun()

    st.markdown("---")
    col_a, col_b, col_c, col_d = st.columns(4)
    for col, text in zip([col_a,col_b,col_c,col_d], ["CKKS Encryption","Cosine Similarity","Rule-Based Decision Analysis","PDF Reports"]):
        col.markdown(f'<div style="text-align:center;padding:10px;border:1px solid #1A2D4A;border-radius:8px;font-size:11px;color:#5A7A9A;letter-spacing:0.06em;">{text}</div>', unsafe_allow_html=True)

    # TenSEAL warning
    if not st.session_state.tenseal_available:
        st.warning("⚠ TenSEAL not installed. Run: `pip install tenseal`  — The app requires TenSEAL for real FHE operations.")


def page_admin_login():
    st.markdown('<div style="max-width:420px;margin:80px auto 0;">', unsafe_allow_html=True)
    if st.button("← Back", key="back_login"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("""
    <div class="bio-card bio-card-glow" style="text-align:center;margin-top:20px;">
        <div style="font-size:40px;margin-bottom:12px;">🔐</div>
        <h2>Admin Access</h2>
        <p style="color:#5A7A9A;font-size:12px;margin-bottom:24px;">Enter your credentials to continue</p>
    </div>
    """, unsafe_allow_html=True)

    pw = st.text_input("Password", type="password", placeholder="Enter admin password")
    if st.button("Authenticate", use_container_width=True):
        if pw == "admin123":
            st.session_state.admin_logged_in = True
            st.session_state.page = "admin"
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.caption("Demo password: `admin123`")
    st.markdown('</div>', unsafe_allow_html=True)


def page_admin():
    if not st.session_state.admin_logged_in:
        st.session_state.page = "admin_login"
        st.rerun()

    # Sidebar nav
    with st.sidebar:
        st.markdown('<div style="display:flex;align-items:center;gap:10px;padding:0 8px;margin-bottom:24px;"><span style="color:#00E5FF;font-size:20px;">🔐</span><span style="font-family:Syne;font-weight:700;font-size:15px;">Admin Panel</span></div>', unsafe_allow_html=True)
        tab = st.radio("Navigation", ["📤 Upload Database", "🗄 View Database", "⚙ System Info"], label_visibility="collapsed")

        st.markdown("---")
        if st.session_state.database:
            st.markdown(f"""
            <div style="padding:10px 12px;background:rgba(0,230,118,0.08);border-radius:8px;
                        border:1px solid rgba(0,230,118,0.2);margin-bottom:12px;">
                <p style="font-size:10px;color:#00E676;text-transform:uppercase;letter-spacing:0.08em;">Database Active</p>
                <p style="font-size:14px;font-weight:600;margin-top:2px;">{len(st.session_state.database)} users</p>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.admin_logged_in = False
            st.session_state.page = "home"
            st.rerun()

    # ── Upload tab ─────────────────────────────────────────────────────────────
    if "Upload" in tab:
        st.title("Upload Dataset")
        st.markdown('<p style="color:#5A7A9A;font-size:13px;margin-bottom:24px;">Upload your biometric JSON dataset to build the encrypted database</p>', unsafe_allow_html=True)

        if not st.session_state.tenseal_available:
            st.error("TenSEAL is not installed. Please run: `pip install tenseal`")
            return

        uploaded = st.file_uploader("Drop JSON dataset here", type=["json"],
                                     help="Format: {userId: {features: [[...]]}")

        st.markdown("""
        <div class="bio-card" style="margin-top:16px;">
            <h3 style="font-size:13px;margin-bottom:10px;">Expected JSON Format</h3>
            <pre style="font-size:11px;color:#00FF41;background:#000;padding:14px;border-radius:8px;overflow-x:auto;line-height:1.6;">{
  "1": {
    "features": [
      [-403.69, -198.48, 77.55, -37.47, 35.11],
      [-370.80, -215.53, 69.66, -22.57, -20.09],
      ...more feature vectors
    ]
  },
  "2": { "features": [...] },
  ...
}</pre>
        </div>
        """, unsafe_allow_html=True)

        if uploaded:
            try:
                data = json.load(uploaded)
                st.success(f"✓ File parsed: {len(data)} users found")

                if st.button("🔐 Build Encrypted Database", use_container_width=True):
                    prog_bar = st.progress(0)
                    status_box = st.empty()
                    log_lines = []
                    log_box = st.empty()

                    def progress_cb(pct, msg):
                        prog_bar.progress(min(pct, 1.0))
                        status_box.markdown(f'<p style="color:#00E5FF;font-size:12px;">⟳ {msg}</p>', unsafe_allow_html=True)
                        log_lines.append(msg)
                        log_box.markdown(
                            '<div class="terminal">' +
                            ''.join(f'<div>{l}</div>' for l in log_lines[-20:]) +
                            '</div>',
                            unsafe_allow_html=True
                        )

                    with st.spinner("Running TenSEAL CKKS encryption..."):
                        enc_db, priv_keys, logs = build_encrypted_database(data, progress_cb)

                    prog_bar.progress(1.0)
                    status_box.empty()

                    st.session_state.database = data
                    st.session_state.encrypted_database = enc_db
                    st.session_state.user_private_keys = priv_keys

                    final_log = '<div class="terminal">' + ''.join(f'<div style="color:{"#00E676" if "✓" in l else "#FF4444" if "✗" in l else "#00FF41"}">{l}</div>' for l in logs) + '</div>'
                    log_box.markdown(final_log, unsafe_allow_html=True)
                    st.success(f"✅ Encrypted database built for {len(enc_db)} users!")

            except Exception as e:
                st.error(f"Error: {e}")

    # ── View DB tab ────────────────────────────────────────────────────────────
    elif "View" in tab:
        st.title("Encrypted Database")
        if not st.session_state.encrypted_database:
            st.warning("No database loaded. Upload a JSON file first.")
            if st.button("Go to Upload"):
                st.rerun()
        else:
            db = st.session_state.encrypted_database
            raw = st.session_state.database
            user_ids = list(db.keys())

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Users", len(user_ids))
            c2.metric("Total Encrypted Vectors", sum(len(db[u]["encrypted_features"]) for u in user_ids))
            c3.metric("Feature Dimensions", len(raw[user_ids[0]]["features"][0]) if raw else "–")

            st.markdown("---")
            rows = ""
            for uid in user_ids[:30]:
                n_templates = len(db[uid]["encrypted_features"])
                feat_dim = len(raw[uid]["features"][0]) if raw else "–"
                rows += f"""<tr>
                    <td style="color:#00E5FF;font-weight:600;">{uid}</td>
                    <td>{n_templates}</td>
                    <td>{feat_dim}</td>
                    <td><span class="badge badge-success">✓ CKKS Encrypted</span></td>
                </tr>"""
            if len(user_ids) > 30:
                rows += f'<tr><td colspan="4" style="text-align:center;color:#5A7A9A;padding:14px;">+{len(user_ids)-30} more users...</td></tr>'

            st.markdown(f"""
            <div class="bio-card" style="padding:0;overflow:hidden;">
                <table class="bio-table"><thead><tr>
                    <th>User ID</th><th>Templates</th><th>Feature Dim</th><th>Status</th>
                </tr></thead><tbody>{rows}</tbody></table>
            </div>
            """, unsafe_allow_html=True)

    # ── System info tab ────────────────────────────────────────────────────────
    elif "System" in tab:
        st.title("System Information")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="bio-card bio-card-glow">
                <h3>Encryption Parameters</h3>
            """, unsafe_allow_html=True)
            for k, v in [("Scheme","CKKS (Approx. HE)"),("Library","TenSEAL"),
                         ("poly_modulus_degree","8192"),("coeff_mod_bit_sizes","[60, 40, 40, 60]"),
                         ("global_scale","2^40"),("Security Level","128-bit")]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:8px 0;
                            border-bottom:1px solid #1A2D4A;font-size:20px;">
                    <span style="color:#5A7A9A;">{k}</span>
                    <span style="color:#00E5FF;font-weight:600;">{v}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="bio-card bio-card-purple">
                <h3 style="color:#7B61FF;">Authentication Rules</h3>
            """, unsafe_allow_html=True)
            for k, v in [("Feature Threshold","≥ 0.999"),("Min Templates","≥ 2"),
                         ("Average Threshold","0.70"),("Similarity Metric","Cosine (HE domain)"),
                         ("Normalization","L2 Unit Vector"),("XAI Rule","N features → N templates")]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:8px 0;
                            border-bottom:1px solid #1A2D4A;font-size:20px;">
                    <span style="color:#5A7A9A;">{k}</span>
                    <span style="color:#7B61FF;font-weight:600;">{v}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        status_color = "#00E676" if st.session_state.tenseal_available else "#FF4444"
        status_text  = "INSTALLED ✓" if st.session_state.tenseal_available else "NOT FOUND ✗"
        st.markdown(f"""
        <div class="bio-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:14px;">TenSEAL Library</span>
                <span style="color:{status_color};font-weight:700;font-size:14px;">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if not st.session_state.tenseal_available:
            st.code("pip install tenseal", language="bash")


def page_user():
    with st.sidebar:
        st.markdown('<div style="display:flex;align-items:center;gap:10px;padding:0 8px;margin-bottom:24px;"><span style="font-size:20px;">👤</span><span style="font-family:Syne;font-weight:700;font-size:15px;">User Portal</span></div>', unsafe_allow_html=True)
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.auth_result = None
            st.rerun()

        st.markdown("---")
        if st.session_state.encrypted_database:
            st.markdown(f"""
            <div style="padding:10px 12px;background:rgba(0,230,118,0.08);border-radius:8px;border:1px solid rgba(0,230,118,0.2);">
                <p style="font-size:10px;color:#00E676;text-transform:uppercase;">DB Active</p>
                <p style="font-size:13px;font-weight:600;">{len(st.session_state.encrypted_database)} users enrolled</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="padding:10px 12px;background:rgba(255,68,68,0.08);border-radius:8px;border:1px solid rgba(255,68,68,0.2);"><p style="font-size:10px;color:#FF4444;text-transform:uppercase;">No DB Loaded</p><p style="font-size:11px;color:#5A7A9A;margin-top:2px;">Admin must upload dataset first</p></div>', unsafe_allow_html=True)

    if st.session_state.auth_result:
        # ── Result view ────────────────────────────────────────────────────────
        result = st.session_state.auth_result
        auth = result["authenticated"]
        xai  = result["xai_struct"]

        if auth:
            st.markdown(f"""
            <div class="result-auth">
                <div style="font-size:64px;margin-bottom:12px;">✅</div>
                <div class="result-title" style="color:#00E676;">AUTHENTICATED</div>
                <p style="color:#5A7A9A;font-size:14px;margin-top:10px;">
                    User ID: <strong style="color:#E2EAF4;">{xai['query_user_id']}</strong> —
                    Best Score: <strong style="color:#00E5FF;">{xai['authentication_rules']['traditional']['best_score']:.4f}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fail">
                <div style="font-size:64px;margin-bottom:12px;">❌</div>
                <div class="result-title" style="color:#FF4444;">AUTHENTICATION FAILED</div>
                <p style="color:#5A7A9A;font-size:14px;margin-top:10px;">
                    User ID: <strong style="color:#E2EAF4;">{xai['query_user_id']}</strong> —
                    Best Score: <strong style="color:#FF4444;">{xai['authentication_rules']['traditional']['best_score']:.4f}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Badges
        rb_ok = xai["authentication_rules"]["rule_based"]["success"]
        tr_ok = xai["authentication_rules"]["traditional"]["authenticated"]
        agree = xai["confidence_metrics"]["methods_agree"]
        st.markdown(f"""
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px;">
            <span class="badge badge-{'success' if rb_ok else 'danger'}">Rule-Based: {'PASS' if rb_ok else 'FAIL'}</span>
            <span class="badge badge-{'success' if tr_ok else 'danger'}">Traditional: {'PASS' if tr_ok else 'FAIL'}</span>
            <span class="badge badge-{'success' if agree else 'warn'}">Methods: {'AGREE' if agree else 'DISAGREE'}</span>
        </div>
        """, unsafe_allow_html=True)

        # Full XAI
        st.markdown("---")
        st.markdown("## 🔍 Decision Analysis")
        render_xai_full(result)

        # Downloads
        st.markdown("---")
        st.markdown("### 📥 Download Reports")

        c1, c2, c3 = st.columns(3)

        with c1:
            xai_json = json.dumps(xai, indent=2, cls=NumpyEncoder)
            st.download_button(
                "⬇ JSON Report",
                data=xai_json,
                file_name=f"XAI_Report_User{xai['query_user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        with c2:
            # Generate PDF — exact notebook function
            try:
                min_req = xai["configuration"]["min_templates_required"]
                with st.spinner("Generating PDF..."):
                    pdf_bytes = generate_comprehensive_xai_pdf(
                        xai,
                        result["template_scores"],
                        result["feature_similarities_all"],
                        xai["query_user_id"],
                        min_req
                    )
                fname = f"Decision_Report_User{xai['query_user_id']}_{xai['authentication_rules']['traditional']['best_score']:.3f}.pdf"
                st.download_button(
                    "⬇ PDF Report (Full)",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.caption("Install: `pip install reportlab`")

        with c3:
            if st.button("← New Authentication", use_container_width=True):
                st.session_state.auth_result = None
                st.rerun()

    else:
        # ── Input view ─────────────────────────────────────────────────────────
        st.title("User Verification")
        st.markdown('<p style="color:#5A7A9A;font-size:13px;margin-bottom:24px;">Submit your feature vector for encrypted biometric authentication via TenSEAL CKKS</p>', unsafe_allow_html=True)

        if not st.session_state.tenseal_available:
            st.error("TenSEAL not installed. Run: `pip install tenseal`")
            return

        if not st.session_state.encrypted_database:
            st.warning("⚠ No database loaded. An admin must upload and encrypt the dataset first.")

        col_form, col_info = st.columns([3, 2], gap="large")

        with col_form:
            st.markdown('<div class="bio-card bio-card-purple">', unsafe_allow_html=True)
            st.markdown("### Authentication Request")

            user_id = st.text_input("User ID", placeholder="e.g. 24",
                                     help="Must exist in the encrypted database")

            # Demo fill button
            if st.button("📋 Use Demo Data (User 24 from notebook)", use_container_width=False):
                st.session_state["_demo_filled"] = True
                st.rerun()

            demo_val = ""
            if st.session_state.get("_demo_filled"):
                demo_val = '[[-403.691303565368, -198.4814013615891, 77.55259193171345, -37.471253039648545, 35.11382073237985], [-370.80670640425615, -215.53142283509877, 69.66043410086661, -22.568455230436133, -20.088267124066007]]'

            feature_text = st.text_area(
                "Feature Vectors (JSON array of arrays)",
                value=demo_val,
                height=160,
                placeholder='[[-403.69, -198.48, 77.55, -37.47, 35.11], [-370.80, -215.53, 69.66, -22.57, -20.09]]',
                help="Paste your raw feature vectors. They will be normalized & encrypted before comparison."
            )

            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🔐 Authenticate via FHE", use_container_width=True):
                if not user_id.strip():
                    st.error("Please enter a User ID")
                elif not st.session_state.encrypted_database:
                    st.error("No encrypted database. Ask admin to upload dataset.")
                elif user_id not in st.session_state.encrypted_database:
                    st.error(f'User ID "{user_id}" not found in encrypted database')
                else:
                    try:
                        query_features = json.loads(feature_text)
                        if not isinstance(query_features, list) or not isinstance(query_features[0], list):
                            raise ValueError("Must be array of arrays")
                    except Exception as e:
                        st.error(f"Invalid feature format: {e}")
                        st.stop()

                    with st.spinner("Running TenSEAL FHE authentication..."):
                        try:
                            result = authenticate_user(
                                user_id,
                                query_features,
                                st.session_state.encrypted_database,
                                st.session_state.user_private_keys
                            )
                            st.session_state.auth_result = result
                            st.session_state["_demo_filled"] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Authentication error: {e}")

        with col_info:
            st.markdown("""
            <div class="bio-card" style="margin-bottom:16px;">
                <h3>🔐 Privacy Guarantee</h3>
                <div style="display:flex;flex-direction:column;gap:10px;margin-top:12px;">
            """, unsafe_allow_html=True)
            for item in [
                "Feature vectors normalized (L2) before encryption",
                "CKKS homomorphic encryption via TenSEAL",
                "Cosine similarity computed in encrypted domain",
                "No plaintext ever leaves your device",
                "Per-user CKKS keypairs (poly_modulus_degree=8192)"
            ]:
                st.markdown(f'<div style="font-size:12px;color:#5A7A9A;display:flex;gap:8px;"><span style="color:#00E676;">✓</span>{item}</div>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="bio-card">
                <h3>📋 Auth Rule (from notebook)</h3>
                <div style="font-size:12px;color:#5A7A9A;line-height:1.8;margin-top:10px;">
                    <div>• Feature threshold: <strong style="color:#FFD700;">≥ 0.999</strong></div>
                    <div>• Min templates: <strong style="color:#00E5FF;">≥ 2</strong> must have a high feature</div>
                    <div>• Average threshold: <strong style="color:#00E5FF;">0.70</strong></div>
                    <div>• Method: <strong style="color:#00E5FF;">CKKS cosine similarity</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "admin_login":
    page_admin_login()
elif st.session_state.page == "admin":
    page_admin()
elif st.session_state.page == "user":
    page_user()
