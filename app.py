"""
=============================================================
  SPAM EMAIL DETECTION SYSTEM  –  Dark UI Streamlit App
  Run with:  streamlit run app.py
=============================================================
"""

import os, sys, joblib, time
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from spam_detector import clean_text, run_training_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DARK THEME CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Orbitron:wght@700;900&display=swap');

/* ── Root variables ── */
:root {
  --bg-deep:    #080c14;
  --bg-panel:   #0d1526;
  --bg-card:    #111d35;
  --bg-input:   #0a1628;
  --accent:     #00d4ff;
  --accent2:    #7b2fff;
  --danger:     #ff3c5a;
  --safe:       #00ffaa;
  --warn:       #ffaa00;
  --text:       #c8d8f0;
  --text-dim:   #4a6080;
  --border:     #1a2d50;
  --glow-cyan:  0 0 20px rgba(0,212,255,0.35);
  --glow-red:   0 0 20px rgba(255,60,90,0.4);
  --glow-green: 0 0 20px rgba(0,255,170,0.35);
}

/* ── Global reset ── */
html, body, [class*="css"] {
  font-family: 'Rajdhani', sans-serif !important;
  background-color: var(--bg-deep) !important;
  color: var(--text) !important;
}

.stApp {
  background: var(--bg-deep) !important;
  background-image:
    radial-gradient(ellipse 80% 40% at 50% -10%, rgba(0,212,255,0.08) 0%, transparent 70%),
    linear-gradient(180deg, var(--bg-deep) 0%, #060a10 100%) !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1100px; }

/* ── Scanline overlay ── */
.stApp::before {
  content: '';
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px
  );
  pointer-events: none; z-index: 9999;
}

/* ── Hero header ── */
.hero {
  text-align: center;
  padding: 2.5rem 1rem 1.5rem;
  position: relative;
}
.hero-badge {
  display: inline-block;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.25em;
  color: var(--accent);
  border: 1px solid rgba(0,212,255,0.3);
  padding: 0.3rem 1rem;
  border-radius: 2px;
  margin-bottom: 1rem;
  background: rgba(0,212,255,0.05);
  animation: pulse-border 2s ease-in-out infinite;
}
@keyframes pulse-border {
  0%,100% { border-color: rgba(0,212,255,0.3); }
  50%      { border-color: rgba(0,212,255,0.8); box-shadow: var(--glow-cyan); }
}
.hero-title {
  font-family: 'Orbitron', monospace;
  font-size: clamp(2rem, 5vw, 3.5rem);
  font-weight: 900;
  letter-spacing: 0.05em;
  margin: 0;
  background: linear-gradient(135deg, #ffffff 0%, var(--accent) 50%, var(--accent2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: none;
  line-height: 1.1;
}
.hero-sub {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.85rem;
  color: var(--text-dim);
  margin-top: 0.6rem;
  letter-spacing: 0.1em;
}

/* ── Status bar ── */
.status-bar {
  display: flex; align-items: center; justify-content: center;
  gap: 1.5rem; padding: 0.6rem 1.5rem;
  background: rgba(0,212,255,0.04);
  border: 1px solid var(--border);
  border-radius: 4px; margin: 1rem 0 2rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.72rem; letter-spacing: 0.08em;
}
.status-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--safe);
  box-shadow: 0 0 6px var(--safe);
  animation: blink 1.4s ease-in-out infinite;
  display: inline-block; margin-right: 0.4rem;
}
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
.status-item { color: var(--text-dim); }
.status-item span { color: var(--accent); }

/* ── Section labels ── */
.section-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.7rem; letter-spacing: 0.2em;
  color: var(--accent); text-transform: uppercase;
  display: flex; align-items: center; gap: 0.5rem;
  margin-bottom: 0.6rem;
}
.section-label::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Input area ── */
.stTextArea textarea {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.9rem !important;
  line-height: 1.6 !important;
  padding: 1rem !important;
  transition: border-color 0.3s, box-shadow 0.3s !important;
  resize: vertical !important;
}
.stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: var(--glow-cyan) !important;
  outline: none !important;
}
.stTextArea textarea::placeholder { color: var(--text-dim) !important; }

/* ── Classify button ── */
.stButton > button {
  width: 100% !important;
  background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(123,47,255,0.12)) !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.15em !important;
  padding: 0.75rem 2rem !important;
  border-radius: 4px !important;
  cursor: pointer !important;
  transition: all 0.25s ease !important;
  text-transform: uppercase !important;
  position: relative !important;
  overflow: hidden !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(123,47,255,0.25)) !important;
  box-shadow: var(--glow-cyan) !important;
  transform: translateY(-1px) !important;
  color: #fff !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result panels ── */
.result-panel {
  border-radius: 8px;
  padding: 1.5rem 2rem;
  margin: 1rem 0;
  position: relative;
  overflow: hidden;
  animation: slide-in 0.4s cubic-bezier(0.16,1,0.3,1);
}
@keyframes slide-in {
  from { opacity: 0; transform: translateY(12px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
.result-spam {
  background: linear-gradient(135deg, rgba(255,60,90,0.12), rgba(255,60,90,0.04));
  border: 1px solid rgba(255,60,90,0.5);
  box-shadow: var(--glow-red), inset 0 0 30px rgba(255,60,90,0.05);
}
.result-ham {
  background: linear-gradient(135deg, rgba(0,255,170,0.10), rgba(0,255,170,0.03));
  border: 1px solid rgba(0,255,170,0.45);
  box-shadow: var(--glow-green), inset 0 0 30px rgba(0,255,170,0.04);
}
.result-label {
  font-family: 'Orbitron', monospace;
  font-size: 1.6rem; font-weight: 900;
  letter-spacing: 0.08em; margin-bottom: 0.4rem;
}
.result-spam .result-label { color: var(--danger); }
.result-ham  .result-label { color: var(--safe); }
.result-desc {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.8rem; color: var(--text-dim); letter-spacing: 0.06em;
}
.result-corner {
  position: absolute; top: 0; right: 0;
  font-size: 4rem; opacity: 0.06; line-height: 1;
  font-family: 'Orbitron', monospace;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
  margin-top: 1rem;
}
.conf-bar-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.68rem; letter-spacing: 0.1em; color: var(--text-dim);
  margin-bottom: 0.3rem; display: flex; justify-content: space-between;
}
.conf-bar-bg {
  height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
}
.conf-bar-fill {
  height: 100%; border-radius: 3px;
  transition: width 0.8s cubic-bezier(0.16,1,0.3,1);
}
.conf-spam { background: linear-gradient(90deg, var(--danger), #ff7a7a); }
.conf-ham  { background: linear-gradient(90deg, var(--safe), #80ffcc); }

/* ── Sample cards ── */
.sample-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.8rem 1rem;
  cursor: pointer;
  transition: all 0.2s ease;
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.95rem; font-weight: 600;
  color: var(--text);
  text-align: left; width: 100%;
  margin-bottom: 0.5rem;
}
.sample-card:hover {
  border-color: var(--accent);
  background: rgba(0,212,255,0.06);
  box-shadow: var(--glow-cyan);
  color: #fff;
  transform: translateX(3px);
}

/* ── Stats panel ── */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem; margin: 1rem 0;
}
.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
  text-align: center;
  transition: border-color 0.2s;
}
.stat-card:hover { border-color: var(--accent); }
.stat-value {
  font-family: 'Orbitron', monospace;
  font-size: 1.6rem; font-weight: 700; color: var(--accent);
}
.stat-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem; letter-spacing: 0.12em; color: var(--text-dim);
  margin-top: 0.2rem; text-transform: uppercase;
}

/* ── Code block ── */
.stCodeBlock { border: 1px solid var(--border) !important; border-radius: 6px !important; }
code { font-family: 'Share Tech Mono', monospace !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  color: var(--text-dim) !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.08em !important;
}
.streamlit-expanderContent {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; opacity: 0.5 !important; }

/* ── Warning / info ── */
.stWarning {
  background: rgba(255,170,0,0.1) !important;
  border: 1px solid rgba(255,170,0,0.4) !important;
  border-radius: 6px !important;
  color: var(--warn) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Footer ── */
.footer {
  text-align: center;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem; letter-spacing: 0.12em;
  color: var(--text-dim); padding: 2rem 0 1rem;
  border-top: 1px solid var(--border);
  margin-top: 2rem;
}
.footer span { color: var(--accent); }

/* ── Column gap ── */
[data-testid="column"] { padding: 0 0.4rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────────
MODEL_PATH = "models/spam_classifier.pkl"
VEC_PATH   = "models/tfidf_vectorizer.pkl"

@st.cache_resource(show_spinner="⚙️  Initialising neural filters …")
def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
        run_training_pipeline()
    return joblib.load(MODEL_PATH), joblib.load(VEC_PATH)

model, vectorizer = load_model()

# ── Session state ─────────────────────────────────────────────────────────────
if "scan_count"  not in st.session_state: st.session_state.scan_count  = 0
if "spam_count"  not in st.session_state: st.session_state.spam_count  = 0
if "last_result" not in st.session_state: st.session_state.last_result = None

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">⬡ AI-POWERED THREAT DETECTION v2.0 ⬡</div>
  <h1 class="hero-title">SPAMSHIELD</h1>
  <p class="hero-sub">INTELLIGENT EMAIL CLASSIFICATION ENGINE · NB + SVM ENSEMBLE</p>
</div>
""", unsafe_allow_html=True)

# ── Status bar ────────────────────────────────────────────────────────────────
ham_count = st.session_state.scan_count - st.session_state.spam_count
st.markdown(f"""
<div class="status-bar">
  <span class="status-item"><span class="status-dot"></span></span>
  <span class="status-item">SYSTEM <span>ONLINE</span></span>
  <span class="status-item">TOTAL SCANS <span>{st.session_state.scan_count}</span></span>
  <span class="status-item">THREATS DETECTED <span>{st.session_state.spam_count}</span></span>
  <span class="status-item">CLEAN MESSAGES <span>{ham_count}</span></span>
  <span class="status-item">MODEL <span>LOADED ✓</span></span>
</div>
""", unsafe_allow_html=True)

# ── Main layout: left = input, right = samples ────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="section-label">▸ EMAIL INPUT TERMINAL</div>', unsafe_allow_html=True)
    email_input = st.text_area(
        label="",
        height=220,
        placeholder="Paste or type email content here...\n\nExample: CONGRATULATIONS! You've won $1,000,000. Click here to claim your prize NOW!",
        key="email_input",
        label_visibility="collapsed",
    )

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        classify_btn = st.button("🔍  SCAN EMAIL", use_container_width=True)
    _, c_clear, _ = st.columns([1, 2, 1])

    # ── Result display ────────────────────────────────────────────────────────
    if classify_btn:
        if not email_input.strip():
            st.warning("⚠  No input detected. Please enter email content.")
        else:
            with st.spinner("Analysing threat vectors …"):
                time.sleep(0.4)
                cleaned  = clean_text(email_input)
                features = vectorizer.transform([cleaned])
                pred     = model.predict(features)[0]

            st.session_state.scan_count += 1
            if pred == 1:
                st.session_state.spam_count += 1

            st.session_state.last_result = {"pred": pred, "cleaned": cleaned,
                                            "text": email_input}
            st.rerun()

    if st.session_state.last_result:
        r = st.session_state.last_result
        pred    = r["pred"]
        cleaned = r["cleaned"]

        if pred == 1:
            st.markdown("""
            <div class="result-panel result-spam">
              <div class="result-corner">⚠</div>
              <div class="result-label">🚨 SPAM DETECTED</div>
              <div class="result-desc">THREAT LEVEL: HIGH · This message matches known spam patterns.</div>
              <div class="conf-bar-wrap">
                <div class="conf-bar-label"><span>SPAM PROBABILITY</span><span>HIGH</span></div>
                <div class="conf-bar-bg"><div class="conf-bar-fill conf-spam" style="width:92%"></div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-panel result-ham">
              <div class="result-corner">✓</div>
              <div class="result-label">✅ CLEAN MESSAGE</div>
              <div class="result-desc">THREAT LEVEL: NONE · This message appears to be legitimate.</div>
              <div class="conf-bar-wrap">
                <div class="conf-bar-label"><span>LEGITIMACY SCORE</span><span>HIGH</span></div>
                <div class="conf-bar-bg"><div class="conf-bar-fill conf-ham" style="width:95%"></div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("⚙  VIEW PROCESSED TOKENS"):
            st.code(cleaned if cleaned else "(empty after cleaning)", language=None)

with right:
    st.markdown('<div class="section-label">▸ QUICK-SCAN SAMPLES</div>', unsafe_allow_html=True)

    samples = {
        "💰  Lottery Prize Scam":    "CONGRATULATIONS! You've won a $5000 lottery prize! Call now to claim your reward. Limited time offer!",
        "💊  Pharma Spam":           "Buy cheap medications online! No prescription needed. Huge discounts on all drugs. Order now!",
        "🏦  Bank Phishing":         "URGENT: Your account has been suspended. Click here immediately to verify your details and restore access.",
        "📅  Work Meeting (Ham)":    "Hi, just a reminder about our team meeting tomorrow at 10 AM. Please bring your project updates.",
        "👋  Friendly Message (Ham)":"Hey! Are you free this weekend? We are planning a barbecue at my place on Saturday.",
        "📦  Delivery Notice (Ham)": "Your package has been delivered to your front door. Please collect it at your earliest convenience.",
    }

    for label, text in samples.items():
        if st.button(label, key=f"sample_{label}", use_container_width=True):
            cleaned  = clean_text(text)
            features = vectorizer.transform([cleaned])
            pred     = model.predict(features)[0]
            st.session_state.scan_count += 1
            if pred == 1:
                st.session_state.spam_count += 1
            st.session_state.last_result = {"pred": pred, "cleaned": cleaned, "text": text}
            st.rerun()

    # ── Live session stats ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">▸ SESSION STATISTICS</div>', unsafe_allow_html=True)
    detection_rate = (
        round(st.session_state.spam_count / st.session_state.scan_count * 100)
        if st.session_state.scan_count > 0 else 0
    )
    st.markdown(f"""
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-value">{st.session_state.scan_count}</div>
        <div class="stat-label">Scanned</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color:var(--danger)">{st.session_state.spam_count}</div>
        <div class="stat-label">Spam</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color:var(--safe)">{detection_rate}%</div>
        <div class="stat-label">Threat Rate</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Model metrics charts ──────────────────────────────────────────────────────
if os.path.exists("outputs/confusion_matrices.png"):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">▸ MODEL PERFORMANCE METRICS</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image("outputs/confusion_matrices.png", use_container_width=True)
    if os.path.exists("outputs/accuracy_comparison.png"):
        with col2:
            st.image("outputs/accuracy_comparison.png", use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  SPAMSHIELD AI · BUILT WITH <span>SCIKIT-LEARN</span> + <span>STREAMLIT</span>
  · NAIVE BAYES + SVM ENSEMBLE · <span>ALL SYSTEMS NOMINAL</span>
</div>
""", unsafe_allow_html=True)
