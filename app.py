import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binom
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SSDI — AI & Creativity",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #0f0f13;
    border-right: 1px solid #2a2a3a;
}
[data-testid="stSidebar"] * {
    color: #e8e8f0 !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #a0a0b8 !important;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.main .block-container {
    background: #0d0d14;
    padding-top: 2rem;
    max-width: 1200px;
}
body, .stApp {
    background-color: #0d0d14;
}

.hero-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(82,130,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-sub {
    color: #7b8db0;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
.hero-team {
    display: inline-block;
    background: rgba(82,130,255,0.12);
    border: 1px solid rgba(82,130,255,0.3);
    border-radius: 6px;
    padding: 0.25rem 0.75rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #6b9fff;
    margin-top: 1rem;
}

.section-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #6b9fff;
    margin: 0 0 0.3rem 0;
    font-weight: 700;
}
.section-subtitle {
    color: #5a6070;
    font-size: 0.82rem;
    margin-bottom: 1.2rem;
    font-style: italic;
}

.var-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
.var-pill {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.pill-cont   { background: rgba(29,158,117,0.15); color: #1D9E75; border: 1px solid rgba(29,158,117,0.35); }
.pill-cat    { background: rgba(83,74,183,0.15);  color: #8b7fff; border: 1px solid rgba(83,74,183,0.35); }
.pill-target { background: rgba(216,90,48,0.15);  color: #e8714a; border: 1px solid rgba(216,90,48,0.35); }
.pill-label  { font-size: 0.68rem; color: #555; font-family: 'DM Sans', sans-serif; font-weight: 400; }

.formula-box {
    background: #0a0a12;
    border: 1px solid #252535;
    border-left: 3px solid #6b9fff;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #a8c0ff;
    margin: 0.75rem 0;
    overflow-x: auto;
}

.badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 0.4rem;
}
.badge-lab  { background: rgba(107,159,255,0.15); color: #6b9fff; border: 1px solid rgba(107,159,255,0.3); }
.badge-pred { background: rgba(216,90,48,0.15);   color: #e8714a; border: 1px solid rgba(216,90,48,0.3); }

.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
    background: #0e0e1a;
    border: 1px solid #1e2030;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    min-width: 110px;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}
.metric-lbl { font-size: 0.7rem; color: #555; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em; }

.hdivider { border: none; border-top: 1px solid #1a1a2a; margin: 1.5rem 0; }

.need-box {
    background: rgba(107,159,255,0.06);
    border: 1px solid rgba(107,159,255,0.15);
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #8090b0;
    margin-bottom: 1rem;
    line-height: 1.6;
}

.result-banner {
    border-radius: 8px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.result-accept { background: rgba(29,158,117,0.12); border: 1px solid rgba(29,158,117,0.35); }
.result-reject { background: rgba(216,90,48,0.12);  border: 1px solid rgba(216,90,48,0.35); }
.result-text { font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; }
.result-accept .result-text { color: #1D9E75; }
.result-reject .result-text { color: #e8714a; }

div[data-testid="stFileUploader"] { background: #0f0f1a; border: 1px dashed #2a2a3a; border-radius: 8px; }
.stSlider > div > div { color: #6b9fff; }
label { color: #9090a8 !important; font-size: 0.82rem !important; }
.stSelectbox > div > div { background: #0f0f1a; border-color: #2a2a3a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#13131f', 'axes.facecolor': '#13131f',
    'axes.edgecolor': '#2a2a3a', 'text.color': '#c0c0d8',
    'axes.labelcolor': '#c0c0d8', 'xtick.color': '#6a6a80',
    'ytick.color': '#6a6a80', 'axes.grid': True,
    'grid.color': '#1e1e2e', 'grid.alpha': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10,
})

C1, C2, C3 = '#1D9E75', '#534AB7', '#D85A30'
CB, CG = '#6b9fff', '#e8714a'

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_clean(file):
    df_raw = pd.read_csv(file, encoding='utf-8', encoding_errors='replace')
    cols = df_raw.columns.tolist()

    hours_map = {
        'Less than 30 min': 0.25, '30 min - 1 hr': 0.75,
        '1 - 2 hrs': 1.5, '2 - 4 hrs': 3.0, 'More than 4 hrs': 5.0
    }
    likert = {
        'Strongly agree': 5, 'Agree': 4, 'Neutral': 3,
        'Disagree': 2, 'Strongly disagree': 1
    }

    df = pd.DataFrame()
    df['hours']     = df_raw[cols[4]].map(hours_map)
    df['writing']   = pd.to_numeric(df_raw[cols[6]],  errors='coerce')
    df['problem']   = pd.to_numeric(df_raw[cols[7]],  errors='coerce')
    df['creative']  = pd.to_numeric(df_raw[cols[8]],  errors='coerce')
    df['cb']        = pd.to_numeric(df_raw[cols[10]], errors='coerce')
    df['cn']        = pd.to_numeric(df_raw[cols[11]], errors='coerce')
    df['conf']      = pd.to_numeric(df_raw[cols[12]], errors='coerce')
    df['feel_less'] = df_raw[cols[13]].map(likert)
    df['harder']    = df_raw[cols[14]].map(likert)
    df['accept']    = pd.to_numeric(df_raw[cols[15]], errors='coerce')
    df['effort_raw']= df_raw[cols[16]].str.strip()
    df['role']      = df_raw[cols[19]].str.strip()
    df['gender']    = df_raw[cols[20]].str.strip()
    df['trained']   = (df_raw[cols[21]].str.strip().str.lower() == 'yes').astype(int)

    df = df.dropna(subset=['cb','cn','conf','feel_less','harder','accept',
                           'hours','writing','problem','creative'])

    df['change']     = df['cn'] - df['cb']
    # FIX: dep uses consistent 1-5 scale — accept is 1-10, so normalise it to 1-5
    df['dep']        = (df['feel_less'] + df['harder'] + (df['accept'] / 2)) / 3
    df['effort_sig'] = df['effort_raw'].str.contains('significantly', case=False, na=False).astype(int)
    df['role_b']     = (df['role'] == 'Student').astype(int)

    def usage_type(row):
        if row['problem'] > row['creative']:   return 'Problem-focused'
        elif row['creative'] > row['problem']: return 'Creative-focused'
        else:                                  return 'Balanced'
    df['usage_type'] = df.apply(usage_type, axis=1)

    return df

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def badge(text, kind='lab'):
    cls = 'badge-lab' if kind == 'lab' else 'badge-pred'
    return f'<span class="badge {cls}">{text}</span>'

def pill(text, kind='cont'):
    return f'<span class="var-pill pill-{kind}">{text}</span>'

def formula_box(formula):
    return f'<div class="formula-box">📐 {formula}</div>'

def result_box(decision, text):
    cls = 'result-accept' if 'ACCEPT' in decision or 'SIGNIFICANT' in decision and 'NOT' not in decision else 'result-reject'
    return f'<div class="result-banner {cls}"><div class="result-text">{decision}</div><div style="color:#8090b0;font-size:0.85rem;margin-top:0.4rem">{text}</div></div>'

def metric_card(val, lbl):
    return f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>'

def show_plot(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data")
    uploaded = st.file_uploader("Upload survey CSV", type=['csv'])
    st.markdown("---")
    st.markdown("### 🧭 Navigate")
    section = st.radio("", [
        "🏠 Overview",
        "📊 A1 — Regression to Mean",
        "📊 A2 — Dummy Regression",
        "📊 A3 — Paired t-test",
        "📊 A4 — Lasso Dependency",
        "📊 A5 — Logistic Effort",
        "📊 A6 — Two-sample t-test",
        "📊 A7 — Two-Way ANOVA",
        "🔮 P1 — Predict Creativity",
        "🔮 P2 — Predict Dependency",
        "📋 Summary",
    ], label_visibility='collapsed')
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#404055;line-height:1.7'>
    <b style='color:#555'>VARIABLE LEGEND</b><br>
    <span style='color:#1D9E75'>■</span> Continuous<br>
    <span style='color:#8b7fff'>■</span> Categorical<br>
    <span style='color:#e8714a'>■</span> Target / Outcome
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🧠 AI's Impact on Human Creativity<br>&amp; Independent Thinking</div>
  <p class="hero-sub">SSDI Research Project — Statistical Analysis Dashboard</p>
  <div class="hero-team">Om Parekh (E031) · Aarav Shah (E056) · Madhav Mehta (E057)</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  NO DATA STATE
# ─────────────────────────────────────────────
if uploaded is None:
    st.info("⬆️ Upload your survey CSV in the sidebar to begin. All analyses will populate automatically.")
    st.markdown("""
    <div class="section-card">
      <div class="section-title">📦 Expected Variables</div>
      <div class="section-subtitle">Columns the app reads from your CSV (by position)</div>
      <hr class="hdivider">
      <b style='color:#6b9fff;font-size:0.8rem'>CONTINUOUS</b>
      <div class="var-grid" style="margin-top:0.5rem">
        <span class="var-pill pill-cont">hours</span>
        <span class="var-pill pill-cont">writing</span>
        <span class="var-pill pill-cont">problem</span>
        <span class="var-pill pill-cont">creative</span>
        <span class="var-pill pill-cont">cb</span>
        <span class="var-pill pill-cont">cn</span>
        <span class="var-pill pill-cont">conf</span>
        <span class="var-pill pill-cont">feel_less</span>
        <span class="var-pill pill-cont">harder</span>
        <span class="var-pill pill-cont">accept</span>
      </div>
      <b style='color:#8b7fff;font-size:0.8rem'>CATEGORICAL</b>
      <div class="var-grid" style="margin-top:0.5rem">
        <span class="var-pill pill-cat">role</span>
        <span class="var-pill pill-cat">gender</span>
        <span class="var-pill pill-cat">trained</span>
        <span class="var-pill pill-cat">effort_raw</span>
        <span class="var-pill pill-cat">usage_type</span>
      </div>
      <b style='color:#e8714a;font-size:0.8rem'>DERIVED / TARGET</b>
      <div class="var-grid" style="margin-top:0.5rem">
        <span class="var-pill pill-target">change</span>
        <span class="var-pill pill-target">dep</span>
        <span class="var-pill pill-target">effort_sig</span>
        <span class="var-pill pill-target">role_b</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
df = load_and_clean(uploaded)
n  = len(df)

# Guard: ensure enough data
if n < 5:
    st.error("Not enough valid rows after cleaning. Please check your CSV column positions.")
    st.stop()

# ══════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════
if section == "🏠 Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", n)
    c2.metric("Students", int((df['role'] == 'Student').sum()))
    c3.metric("Professionals", int((df['role'] == 'Working professional').sum()))
    c4.metric("AI-Trained", int(df['trained'].sum()))

    st.markdown("""
    <div class="section-card">
      <div class="section-title">📦 Variable Dictionary</div>
      <div class="section-subtitle">All variables used across the 7 analyses and 2 predictions</div>
      <hr class="hdivider">
    """, unsafe_allow_html=True)

    var_data = {
        "Variable": ["hours","writing","problem","creative","cb","cn","conf",
                     "feel_less","harder","accept","role","gender","trained",
                     "effort_raw","usage_type","change","dep","effort_sig","role_b"],
        "Type": ["Continuous","Continuous","Continuous","Continuous","Continuous","Continuous","Continuous",
                 "Continuous","Continuous","Continuous",
                 "Categorical","Categorical","Categorical","Categorical","Categorical",
                 "Target","Target","Target","Target"],
        "Description": [
            "Daily AI usage in hours (0.25 – 5.0)",
            "AI use for writing tasks (1–10 scale)",
            "AI use for problem-solving (1–10 scale)",
            "AI use for creativity tasks (1–10 scale)",
            "Self-rated creativity BEFORE AI (1–10)",
            "Self-rated creativity NOW (1–10)",
            "Confidence in own thinking (1–5)",
            "Feels less creative using AI (Likert 1–5)",
            "Thinks independently harder now (Likert 1–5)",
            "Accepts AI outputs without checking (1–10)",
            "Student / Working professional",
            "Gender identity",
            "Received AI training (Yes=1, No=0)",
            "Raw effort-reduction response text",
            "Derived: Problem-focused / Creative-focused / Balanced",
            "cn − cb (creativity change score)",
            "Mean dependency index (feel_less, harder, accept/2) / 3",
            "1 if effort 'significantly' reduced, else 0",
            "1 if Student, 0 if Professional"
        ],
        "Used In": [
            "A1,A2,A4,A5,P1","A4,A5","A4,A5,P1","A4,A5,P1","A1,P1","A2,A3,A5–A7,P1",
            "A2,A6,P1","A4","A4","A4","A6,A7","—","A2,A5,A6,P1","A5","A7",
            "A1,A3","A4,P2","A5","A2"
        ]
    }
    vdf = pd.DataFrame(var_data)

    # FIX: applymap deprecated in pandas >= 2.1; use map() instead
    def color_type(val):
        if val == "Continuous":  return "color: #1D9E75"
        if val == "Categorical": return "color: #8b7fff"
        return "color: #e8714a"

    try:
        styled = vdf.style.map(color_type, subset=['Type'])
    except AttributeError:
        # Fallback for older pandas versions
        styled = vdf.style.applymap(color_type, subset=['Type'])

    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
      <div class="section-title">🗺️ Analysis Map</div>
      <hr class="hdivider">
    """, unsafe_allow_html=True)
    map_data = {
        "Code":   ["A1","A2","A3","A4","A5","A6","A7","P1","P2"],
        "Question": [
            "Does high baseline creativity predict decline?",
            "Does training/role/hours predict current creativity?",
            "Has creativity meaningfully changed post-AI?",
            "What drives AI dependency scores?",
            "What predicts effort reduction?",
            "Does training boost confidence by role?",
            "Does usage type × role predict creativity?",
            "Predict current creativity from profile",
            "Are students more AI-dependent?",
        ],
        "Method": [
            "One-Way ANOVA + OLS","Multiple OLS (dummies)","Shapiro + Levene + Paired t + 2-prop Z",
            "Lasso Regression","Logistic Regression","Two-sample t-test",
            "Two-Way ANOVA + Tukey","Multiple OLS + 95% CI","Binomial MLE + 2-prop Z"
        ],
        "Lab Ref": ["8+10","10","9+7","10","11","6","8","10","2+7"],
    }
    st.dataframe(pd.DataFrame(map_data), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  A1
# ══════════════════════════════════════════════
elif section == "📊 A1 — Regression to Mean":
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">A1 — Does high baseline creativity predict a decline?</div>
      {badge("Lab 8 — One-Way ANOVA", "lab")} {badge("Lab 10 — OLS Regression", "lab")}
      <div class="need-box">
        <b>Research Need:</b> Test if people who rated themselves highly creative <i>before</i> AI show
        greater declines (regression-to-mean effect), while low scorers improve.
      </div>
      <b style='color:#9090a8;font-size:0.8rem'>VARIABLES USED</b>
      <div class="var-grid">
        {pill('cb','cont')} <span class="pill-label">Predictor — Creativity Before</span>
        {pill('change','target')} <span class="pill-label">Outcome — cn minus cb</span>
        {pill('group','cat')} <span class="pill-label">Tertile group (Low/Mid/High)</span>
      </div>
      {formula_box("change ~ C(group)  [ANOVA]     change ~ cb  [OLS]")}
    </div>
    """, unsafe_allow_html=True)

    t33, t67 = df['cb'].quantile(0.33), df['cb'].quantile(0.67)
    bot = df[df['cb'] <= t33]['change']
    mid = df[(df['cb'] > t33) & (df['cb'] <= t67)]['change']
    top = df[df['cb'] > t67]['change']

    df_a1 = df.copy()
    df_a1['group'] = 'Mid'
    df_a1.loc[df['cb'] <= t33, 'group'] = 'Low'
    df_a1.loc[df['cb'] >  t67, 'group'] = 'High'

    fit_anova = smf.ols('change ~ C(group)', data=df_a1).fit()
    anova_table = sm.stats.anova_lm(fit_anova, typ=1)
    fit_a1  = smf.ols('change ~ cb', data=df).fit()
    slope   = fit_a1.params['cb']
    pval    = fit_a1.pvalues['cb']

    c1, c2, c3 = st.columns(3)
    c1.metric("OLS Slope", f"{slope:.3f}", help="Negative = high baseline → decline")
    c2.metric("p-value", f"{pval:.4f}", delta="Significant" if pval < 0.05 else "Not sig")
    c3.metric("R²", f"{fit_a1.rsquared:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [C1 if c <= t33 else C3 if c > t67 else C2 for c in df['cb']]
    axes[0].scatter(df['cb'], df['change'], c=colors, s=60, alpha=0.85, zorder=3)
    x_line = np.linspace(df['cb'].min(), df['cb'].max(), 100)
    axes[0].plot(x_line, fit_a1.params['Intercept'] + slope * x_line,
                 color=CB, lw=2, label=f'slope={slope:.2f}')
    axes[0].axhline(0, color='#444', lw=0.8)
    axes[0].set(xlabel='Creativity Before (cb)', ylabel='Change (cn−cb)', title='A1 — Baseline vs Change')
    axes[0].legend()

    group_labels = [f'Low (≤{t33:.0f})', 'Mid', f'High (>{t67:.0f})']
    group_means  = [bot.mean(), mid.mean(), top.mean()]
    axes[1].bar(group_labels, group_means, color=[C1, C2, C3], width=0.45)
    axes[1].axhline(0, color='#888', lw=0.8)
    axes[1].set(ylabel='Mean creativity change', title='A1 — Mean change by baseline group')
    plt.tight_layout()
    show_plot(fig)

    with st.expander("📋 ANOVA Table"):
        st.dataframe(anova_table.round(4))

    sig = pval < 0.05
    st.markdown(result_box(
        "✅ SIGNIFICANT (p<0.05)" if sig else "❌ NOT SIGNIFICANT",
        f"OLS slope = {slope:.3f}, p = {pval:.4f}. Every +1 in baseline → {abs(slope):.2f} drop in change score."
    ), unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  A2
# ══════════════════════════════════════════════
elif section == "📊 A2 — Dummy Regression":
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">A2 — Does training, role or hours predict current creativity?</div>
      {badge("Lab 10 — Multiple OLS with Dummy Variables", "lab")}
      <div class="need-box">
        <b>Research Need:</b> Identify which background factor (daily usage, role, AI training)
        most significantly predicts a respondent's current creativity score.
      </div>
      <div class="var-grid">
        {pill('hours','cont')} {pill('role_b','cat')} {pill('trained','cat')}
        <span class="pill-label"> Predictors</span> &nbsp;
        {pill('cn','target')} <span class="pill-label"> Outcome</span>
      </div>
      {formula_box("cn ~ hours + role_b + trained")}
    </div>
    """, unsafe_allow_html=True)

    fit_a2 = smf.ols('cn ~ hours + role_b + trained', data=df).fit()

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{fit_a2.rsquared:.3f}")
    c2.metric("F-stat", f"{fit_a2.fvalue:.2f}")
    c3.metric("p (model)", f"{fit_a2.f_pvalue:.4f}")

    coef_df = pd.DataFrame({
        'Variable':    fit_a2.params.index.tolist(),
        'Coefficient': fit_a2.params.values.round(4),
        'p-value':     fit_a2.pvalues.values.round(4),
        'Significant': ['✅ YES' if p < 0.05 else '—' for p in fit_a2.pvalues]
    })
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels  = fit_a2.params.index.tolist()
    coefs   = list(fit_a2.params)
    pvals   = list(fit_a2.pvalues)
    bcolors = [CB if p < 0.05 else '#2a2a3a' for p in pvals]
    axes[0].bar(labels, coefs, color=bcolors)
    axes[0].axhline(0, color='#888', lw=0.7)
    axes[0].set(title='A2 — Regression Coefficients\n(blue = significant)', ylabel='Coefficient')

    bp = axes[1].boxplot(
        [df[df['trained'] == 0]['cn'], df[df['trained'] == 1]['cn']],
        labels=['Not Trained', 'AI Trained'],
        patch_artist=True
    )
    bp['boxes'][0].set_facecolor('#2a1a1a'); bp['boxes'][0].set_edgecolor(C3)
    bp['boxes'][1].set_facecolor('#0e2a1a'); bp['boxes'][1].set_edgecolor(C1)
    axes[1].set(ylabel='Creativity Now (cn)', title='A2 — Current Creativity by Training')
    plt.tight_layout()
    show_plot(fig)

    with st.expander("📋 Full OLS Summary"):
        st.text(str(fit_a2.summary()))

# ══════════════════════════════════════════════
#  A3
# ══════════════════════════════════════════════
elif section == "📊 A3 — Paired t-test":
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">A3 — Has creativity meaningfully changed after AI adoption?</div>
      {badge("Lab 9 — Shapiro-Wilk + Levene", "lab")} {badge("Lab 7 — Paired t-test + 2-prop Z", "lab")}
      <div class="need-box">
        <b>Research Need:</b> Test whether the before/after creativity difference is statistically
        significant, and whether more people improved than declined.
      </div>
      <div class="var-grid">
        {pill('cb','cont')} {pill('cn','cont')} <span class="pill-label"> Paired measurements</span>
        &nbsp; {pill('change','target')} <span class="pill-label"> Derived outcome</span>
      </div>
      {formula_box("H₀: μ(cn) = μ(cb)   vs   Hₐ: μ(cn) ≠ μ(cb)")}
    </div>
    """, unsafe_allow_html=True)

    d           = df['cn'] - df['cb']
    W_sw, p_sw  = stats.shapiro(d)
    W_lev,p_lev = stats.levene(df['cb'], df['cn'])
    t_pair, p_pair = stats.ttest_rel(df['cn'], df['cb'])
    df_clean    = df[df['change'] >= -3]
    t_cl, p_cl  = stats.ttest_rel(df_clean['cn'], df_clean['cb'])
    pos = int((df['change'] > 0).sum())
    neg = int((df['change'] < 0).sum())
    total_nonzero = pos + neg

    # FIX: correct one-sample 2-prop Z — tests if proportion improving > 0.5
    if total_nonzero > 0:
        z_a3, p_a3 = proportions_ztest(pos, total_nonzero, value=0.5, alternative='larger')
    else:
        z_a3, p_a3 = 0.0, 1.0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Assumption Tests**")
        st.markdown(f"""
| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Shapiro-Wilk | {W_sw:.3f} | {p_sw:.4f} | {"✅ Normal" if p_sw > 0.05 else "⚠️ Not normal"} |
| Levene | {W_lev:.3f} | {p_lev:.4f} | {"✅ Equal var" if p_lev > 0.05 else "⚠️ Unequal var"} |
        """)
    with col2:
        st.markdown("**Test Results**")
        st.markdown(f"""
| Test | t/z | p-value | Decision |
|------|-----|---------|----------|
| Paired t-test | {t_pair:.3f} | {p_pair:.4f} | {"Reject H₀" if p_pair < 0.05 else "Accept H₀"} |
| Clean sample t | {t_cl:.3f} | {p_cl:.4f} | {"Reject H₀" if p_cl < 0.05 else "Accept H₀"} |
| 2-prop Z | {z_a3:.3f} | {p_a3:.4f} | {"Reject H₀" if p_a3 < 0.05 else "Accept H₀"} |
        """)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    vals       = df['change'].value_counts().sort_index()
    bar_colors = [C3 if v < 0 else C1 if v > 0 else '#888' for v in vals.index]
    axes[0].bar(vals.index, vals.values, color=bar_colors, width=0.7)
    axes[0].axvline(0, color='#888', lw=1)
    axes[0].axvline(df['change'].mean(), color=CB, linestyle='--', lw=1.5,
                    label=f'Mean={df["change"].mean():.2f}')
    axes[0].set(xlabel='Creativity change (cn−cb)', ylabel='Count', title='A3 — Distribution of Changes')
    axes[0].legend()

    x = np.arange(len(df))
    axes[1].plot(x, df['cb'].values, 'o-', color=CB, label='Before (cb)', lw=1.2, ms=4, alpha=0.8)
    axes[1].plot(x, df['cn'].values, 's-', color=C1, label='Now (cn)',    lw=1.2, ms=4, alpha=0.8)
    for i in range(len(df)):
        c = C3 if df['change'].iloc[i] < 0 else '#2a5a3a'
        axes[1].plot([i, i], [df['cb'].iloc[i], df['cn'].iloc[i]], color=c, lw=0.9, alpha=0.5)
    axes[1].set(xlabel='Respondent', ylabel='Score', title='A3 — Before vs Now per Person')
    axes[1].legend()
    plt.tight_layout()
    show_plot(fig)

    st.markdown(result_box(
        f"✅ {pos} IMPROVED · {neg} DECLINED" if p_a3 < 0.05 else f"❌ NOT SIGNIFICANT",
        f"2-prop Z = {z_a3:.3f}, p = {p_a3:.4f}. Mean before={df['cb'].mean():.2f}, after={df['cn'].mean():.2f}."
    ), unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  A4
# ══════════════════════════════════════════════
elif section == "📊 A4 — Lasso Dependency":
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">A4 — What drives AI dependency scores?</div>
      {badge("Lab 10 — Lasso Regression", "lab")}
      <div class="need-box">
        <b>Research Need:</b> Use Lasso's automatic variable selection to identify which predictors
        truly drive the composite dependency index — and which can be zeroed out.
      </div>
      <div class="var-grid">
        {pill('hours','cont')} {pill('writing','cont')} {pill('problem','cont')} {pill('creative','cont')}
        {pill('feel_less','cont')} {pill('harder','cont')} {pill('accept','cont')} {pill('conf','cont')}
        <span class="pill-label"> Predictors (standardised)</span>
        &nbsp; {pill('dep','target')} <span class="pill-label"> Outcome</span>
      </div>
      {formula_box("dep ~ StandardScaler([hours, writing, problem, creative, feel_less, harder, accept, conf])")}
    </div>
    """, unsafe_allow_html=True)

    feats4 = ['hours','writing','problem','creative','feel_less','harder','accept','conf']
    sc4    = StandardScaler()
    X4     = sc4.fit_transform(df[feats4])
    y4     = df['dep'].values

    alpha_val = st.slider("Lasso alpha (regularisation)", 0.001, 0.5, 0.05, 0.005)
    lasso = Lasso(alpha=alpha_val, max_iter=10000)
    lasso.fit(X4, y4)

    r2     = lasso.score(X4, y4)
    kept   = [f for f, c in zip(feats4, lasso.coef_) if abs(c) > 1e-4]
    zeroed = [f for f, c in zip(feats4, lasso.coef_) if abs(c) <= 1e-4]

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2:.4f}")
    c2.metric("Kept Variables", len(kept))
    c3.metric("Zeroed Out", len(zeroed))

    lasso_df = pd.DataFrame({'Variable': feats4, 'Coefficient': lasso.coef_.round(4)})
    lasso_df['Status'] = lasso_df['Coefficient'].apply(lambda x: '✅ KEPT' if abs(x) > 1e-4 else '⬜ zeroed')
    st.dataframe(lasso_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    bcolors = [C2 if abs(c) > 1e-4 else '#2a2a3a' for c in lasso.coef_]
    ax.barh(feats4, lasso.coef_, color=bcolors, height=0.55)
    ax.axvline(0, color='#888', lw=0.8)
    ax.set(xlabel='Lasso Coefficient (standardised)', title=f'A4 — Lasso Variable Selection (α={alpha_val})')
    for bar_patch, val in zip(ax.patches, lasso.coef_):
        if abs(val) > 1e-4:
            ax.text(val + 0.005 * (1 if val >= 0 else -1),
                    bar_patch.get_y() + bar_patch.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)
    kept_p  = mpatches.Patch(color=C2, label='Kept')
    zero_p  = mpatches.Patch(color='#2a2a3a', label='Zeroed', linewidth=1, edgecolor='#555')
    ax.legend(handles=[kept_p, zero_p])
    plt.tight_layout()
    show_plot(fig)

# ══════════════════════════════════════════════
#  A5
# ══════════════════════════════════════════════
elif section == "📊 A5 — Logistic Effort":
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">A5 — What predicts whether AI significantly reduced effort?</div>
      {badge("Lab 11 — Logistic Regression", "lab")}
      <div class="need-box">
        <b>Research Need:</b> Binary classification — predict whether a user reports that AI
        "significantly" reduced their effort, using usage and profile variables.
      </div>
      <div class="var-grid">
        {pill('hours','cont')} {pill('writing','cont')} {pill('problem','cont')} {pill('creative','cont')} {pill('trained','cat')}
        <span class="pill-label"> Predictors</span>
        &nbsp; {pill('effort_sig','target')} <span class="pill-label"> Binary Outcome (0/1)</span>
      </div>
      {formula_box("P(effort_sig=1) = σ(β₀ + β₁·hours + β₂·writing + β₃·problem + β₄·creative + β₅·trained)")}
    </div>
    """, unsafe_allow_html=True)

    feats5 = ['hours','writing','problem','creative','trained']
    sc5    = StandardScaler()
    X5     = sc5.fit_transform(df[feats5])
    y5     = df['effort_sig'].values

    # Guard: need both classes to train logistic regression
    if len(np.unique(y5)) < 2:
        st.warning("⚠️ Only one class present in effort_sig — logistic regression requires both 0 and 1 values.")
        st.stop()

    lr5    = LogisticRegression(max_iter=1000, random_state=42)
    lr5.fit(X5, y5)
    ors5   = np.exp(lr5.coef_[0])
    y_pred = lr5.predict(X5)
    cm     = confusion_matrix(y5, y_pred)
    acc    = accuracy_score(y5, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc * 100:.1f}%")
    c2.metric("Samples (effort_sig=1)", int(y5.sum()))
    c3.metric("Samples (effort_sig=0)", int(len(y5) - y5.sum()))

    or_df = pd.DataFrame({
        'Variable':   feats5,
        'Log-Odds':   lr5.coef_[0].round(4),
        'Odds Ratio': ors5.round(3),
        'Effect':     ['↑ more likely' if o > 1 else '↓ less likely' for o in ors5]
    })
    st.dataframe(or_df, use_container_width=True, hide_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bcolors = [C1 if o > 1 else C3 for o in ors5]
    axes[0].barh(feats5, ors5, color=bcolors, height=0.5)
    axes[0].axvline(1, color=CB, lw=1, linestyle='--')
    axes[0].set(xlabel='Odds Ratio', title='A5 — Logistic Regression Odds Ratios')
    for i, val in enumerate(ors5):
        axes[0].text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=9)

    # FIX: ensure confusion matrix is always 2x2 even if one class is predicted
    cm_full = np.zeros((2, 2), dtype=int)
    for i in range(min(cm.shape[0], 2)):
        for j in range(min(cm.shape[1], 2)):
            cm_full[i, j] = cm[i, j]
    axes[1].imshow(cm_full, cmap='Blues', vmin=0)
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Pred: No', 'Pred: Yes'])
    axes[1].set_yticklabels(['Actual: No', 'Actual: Yes'])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm_full[i, j]), ha='center', va='center',
                         fontsize=18, fontweight='bold', color='white')
    axes[1].set_title(f'A5 — Confusion Matrix (acc={acc * 100:.0f}%)')
    plt.tight_layout()
    show_plot(fig)

# ══════════════════════════════════════════════
#  A6
# ══════════════════════════════════════════════
elif section == "📊 A6 — Two-sample t-test":
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">A6 — Does AI training boost confidence differently for students vs professionals?</div>
      {badge("Lab 6 — Two-sample Independent t-test (Pooled)", "lab")}
      <div class="need-box">
        <b>Research Need:</b> Test whether the confidence gain from AI training differs between
        student and professional groups (separate pooled t-tests per role).
      </div>
      <div class="var-grid">
        {pill('conf','cont')} <span class="pill-label"> Outcome variable</span>
        &nbsp; {pill('role','cat')} {pill('trained','cat')} <span class="pill-label"> Grouping variables</span>
      </div>
      {formula_box("H₀: μ(trained) = μ(untrained)   within each role group   [equal_var=True]")}
    </div>
    """, unsafe_allow_html=True)

    stud_tr = df[(df['role'] == 'Student')              & (df['trained'] == 1)]['conf']
    stud_un = df[(df['role'] == 'Student')              & (df['trained'] == 0)]['conf']
    prof_tr = df[(df['role'] == 'Working professional') & (df['trained'] == 1)]['conf']
    prof_un = df[(df['role'] == 'Working professional') & (df['trained'] == 0)]['conf']

    # FIX: Guard against groups with < 2 samples before t-test
    def safe_ttest(a, b):
        if len(a) < 2 or len(b) < 2:
            return float('nan'), float('nan')
        return stats.ttest_ind(a, b, equal_var=True)

    t_s, p_s = safe_ttest(stud_tr, stud_un)
    t_p, p_p = safe_ttest(prof_tr, prof_un)

    def fmt(v):
        return f"{v:.4f}" if not np.isnan(v) else "N/A (insufficient data)"

    res_df = pd.DataFrame({
        'Group':           ['Students', 'Professionals'],
        'Trained Mean':    [f"{stud_tr.mean():.3f}" if len(stud_tr) > 0 else "N/A",
                            f"{prof_tr.mean():.3f}" if len(prof_tr) > 0 else "N/A"],
        'Untrained Mean':  [f"{stud_un.mean():.3f}" if len(stud_un) > 0 else "N/A",
                            f"{prof_un.mean():.3f}" if len(prof_un) > 0 else "N/A"],
        'Training Effect': [f"{(stud_tr.mean()-stud_un.mean()):.3f}" if len(stud_tr)>0 and len(stud_un)>0 else "N/A",
                            f"{(prof_tr.mean()-prof_un.mean()):.3f}" if len(prof_tr)>0 and len(prof_un)>0 else "N/A"],
        't-stat':    [fmt(t_s), fmt(t_p)],
        'p-value':   [fmt(p_s), fmt(p_p)],
        'Decision':  [
            ('Reject H₀' if p_s < 0.05 else 'Accept H₀') if not np.isnan(p_s) else 'N/A',
            ('Reject H₀' if p_p < 0.05 else 'Accept H₀') if not np.isnan(p_p) else 'N/A'
        ]
    })
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    group_data = [stud_tr, stud_un, prof_tr, prof_un]
    groups     = ['Students\nTrained', 'Students\nUntrained', 'Profs\nTrained', 'Profs\nUntrained']
    bcolors    = [C2, '#2a233a', C1, '#0e2a1a']
    for i, (grp_d, grp_lbl, bc) in enumerate(zip(group_data, groups, bcolors)):
        if len(grp_d) > 0:
            m = grp_d.mean()
            axes[0].bar(grp_lbl, m, color=bc, width=0.55)
            axes[0].text(i, m + 0.1, f'{m:.2f}\n(n={len(grp_d)})', ha='center', fontsize=9)
    axes[0].set_ylim(0, 5.5)
    axes[0].set(ylabel='Mean Confidence (1–5)', title='A6 — Confidence by Role & Training')

    gaps = []
    gap_labels = []
    for role, tr, un in [('Students', stud_tr, stud_un), ('Professionals', prof_tr, prof_un)]:
        if len(tr) > 0 and len(un) > 0:
            gaps.append(tr.mean() - un.mean())
            gap_labels.append(role)
    if gaps:
        axes[1].bar(gap_labels, gaps, color=[C1] * len(gaps), width=0.4)
        axes[1].axhline(0, color='#888', lw=0.8)
        for i, g in enumerate(gaps):
            axes[1].text(i, g + 0.03 * (1 if g >= 0 else -1),
                         f'{g:+.2f}', ha='center', fontsize=12, fontweight='bold', color=C1)
    axes[1].set(ylabel='Confidence Gain from Training', title='A6 — Training Effect by Role')
    plt.tight_layout()
    show_plot(fig)

# ══════════════════════════════════════════════
#  A7
# ══════════════════════════════════════════════
elif section == "📊 A7 — Two-Way ANOVA":
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">A7 — Does usage type × role jointly predict creativity?</div>
      {badge("Lab 8 — Two-Way ANOVA + Tukey HSD", "lab")}
      <div class="need-box">
        <b>Research Need:</b> Test main effects of AI usage type and role, plus their interaction
        effect, on current creativity. Post-hoc Tukey to identify which usage types differ.
      </div>
      <div class="var-grid">
        {pill('usage_type','cat')} {pill('role','cat')} <span class="pill-label"> Factors (IVs)</span>
        &nbsp; {pill('cn','target')} <span class="pill-label"> Outcome (DV)</span>
      </div>
      {formula_box("cn ~ C(usage_type) + C(role) + C(usage_type):C(role)")}
    </div>
    """, unsafe_allow_html=True)

    df_a7 = df[df['role'].isin(['Student', 'Working professional'])].copy()

    # FIX: Need at least 2 unique usage types and 2 roles; fall back to simpler model if needed
    unique_ut = df_a7['usage_type'].nunique()
    unique_rl = df_a7['role'].nunique()

    try:
        if unique_ut > 1 and unique_rl > 1:
            fit_a7 = smf.ols('cn ~ C(usage_type) + C(role) + C(usage_type):C(role)', data=df_a7).fit()
        else:
            fit_a7 = smf.ols('cn ~ C(usage_type)', data=df_a7).fit()
        anova_table = sm.stats.anova_lm(fit_a7, typ=2)
    except Exception as e:
        st.error(f"ANOVA could not be computed: {e}")
        st.stop()

    st.markdown("**Two-Way ANOVA Table**")
    st.dataframe(anova_table.round(4), use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    usage_means = df_a7.groupby('usage_type')['cn'].mean()
    usage_n     = df_a7.groupby('usage_type')['cn'].count()
    ucolors     = {'Balanced': '#555577', 'Creative-focused': C2, 'Problem-focused': C1}

    for i, (ut, mean) in enumerate(usage_means.items()):
        axes[0].bar(ut, mean, color=ucolors.get(ut, '#555'), width=0.5)
        axes[0].text(i, mean + 0.1, f'{mean:.2f}\n(n={usage_n[ut]})', ha='center', fontsize=9)
    axes[0].set(ylabel='Mean Creativity Now (cn)', title='A7 — Creativity by Usage Type', ylim=(0, 11))

    for role, style, color in [('Student', 'o-', C2), ('Working professional', 's--', C1)]:
        subset = df_a7[df_a7['role'] == role].groupby('usage_type')['cn'].mean()
        if len(subset) > 0:
            axes[1].plot(subset.index, subset.values, style, color=color, label=role, lw=2, ms=8)
    axes[1].set(ylabel='Mean cn', title='A7 — Interaction: Usage Type × Role')
    axes[1].legend()
    plt.tight_layout()
    show_plot(fig)

    # FIX: Tukey requires ≥ 2 groups with ≥ 2 observations each
    try:
        if unique_ut >= 2:
            comp  = mc.MultiComparison(df_a7['cn'], df_a7['usage_type'])
            tukey = comp.tukeyhsd()
            with st.expander("📋 Tukey HSD Post-hoc"):
                st.text(str(tukey.summary()))
    except Exception as e:
        st.info(f"Tukey HSD not available: {e}")

# ══════════════════════════════════════════════
#  P1 — PREDICTION
# ══════════════════════════════════════════════
elif section == "🔮 P1 — Predict Creativity":
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">P1 — Predict Current Creativity from a Person's Profile</div>
      {badge("Lab 10 — Multiple OLS + 95% CI", "lab")} {badge("INTERACTIVE PREDICTION", "pred")}
      <div class="need-box">
        <b>Research Need:</b> Given a person's baseline creativity, usage habits and training status,
        predict their current creativity score with a 95% confidence interval.
      </div>
      <div class="var-grid">
        {pill('cb','cont')} {pill('hours','cont')} {pill('creative','cont')} {pill('problem','cont')} {pill('conf','cont')} {pill('trained','cat')}
        <span class="pill-label"> Predictors</span>
        &nbsp; {pill('cn','target')} <span class="pill-label"> Outcome</span>
      </div>
      {formula_box("cn ~ cb + hours + creative + problem + conf + trained")}
    </div>
    """, unsafe_allow_html=True)

    fit_p1 = smf.ols('cn ~ cb + hours + creative + problem + conf + trained', data=df).fit()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{fit_p1.rsquared:.3f}")
    c2.metric("F-stat", f"{fit_p1.fvalue:.2f}")
    c3.metric("p (model)", f"{fit_p1.f_pvalue:.5f}")
    c4.metric("RMSE", f"{np.sqrt(fit_p1.mse_resid):.3f}")

    st.markdown("---")
    st.markdown("### 🎯 Interactive Predictor")
    st.markdown("Enter your profile below to predict current creativity score:")

    col1, col2 = st.columns(2)
    with col1:
        p_cb       = st.slider("Creativity BEFORE AI (cb)", 1, 10, 7)
        p_hours    = st.select_slider(
            "Daily AI usage (hours)", [0.25, 0.75, 1.5, 3.0, 5.0], value=1.5,
            format_func=lambda x: {0.25: "<30min", 0.75: "30m-1hr", 1.5: "1-2hr", 3.0: "2-4hr", 5.0: ">4hr"}[x]
        )
        p_creative = st.slider("AI use for creativity tasks (1–10)", 1, 10, 5)
    with col2:
        p_problem  = st.slider("AI use for problem-solving (1–10)", 1, 10, 5)
        p_conf     = st.slider("Confidence in own thinking (1–5)", 1, 5, 3)
        p_trained  = st.radio("Received AI training?", ["No", "Yes"], horizontal=True)

    p_trained_bin = 1 if p_trained == "Yes" else 0
    new_data = pd.DataFrame([{
        'cb': p_cb, 'hours': p_hours, 'creative': p_creative,
        'problem': p_problem, 'conf': p_conf, 'trained': p_trained_bin
    }])

    pred_frame = fit_p1.get_prediction(new_data).summary_frame(alpha=0.05)
    pred_val   = pred_frame['mean'].iloc[0]
    ci_lo      = pred_frame['mean_ci_lower'].iloc[0]
    ci_hi      = pred_frame['mean_ci_upper'].iloc[0]
    change_exp = pred_val - p_cb
    direction  = "📈 IMPROVE" if change_exp > 0.3 else "📉 DECLINE" if change_exp < -0.3 else "➡️ STABLE"

    st.markdown(f"""
    <div style='background:#0e1e2e;border:1px solid #1e3a5f;border-radius:10px;padding:1.5rem;margin-top:1rem'>
      <div style='font-family:Space Mono,monospace;font-size:0.8rem;color:#5a7aaa;margin-bottom:0.5rem'>PREDICTION RESULT</div>
      <div style='font-family:Space Mono,monospace;font-size:2.2rem;font-weight:700;color:#fff'>{pred_val:.2f}</div>
      <div style='color:#5a7aaa;font-size:0.85rem;margin:0.3rem 0'>95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]</div>
      <div style='font-size:1rem;color:#6b9fff;margin-top:0.5rem'>{direction}  &nbsp;|&nbsp;  Expected change from baseline: {change_exp:+.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.errorbar(0, pred_val, yerr=[[pred_val - ci_lo], [ci_hi - pred_val]],
                fmt='o', color=CB, markersize=14, capsize=12, lw=2.5)
    ax.axhline(df['cn'].mean(), color='#555', linestyle='--', lw=1.2,
               label=f"Sample mean={df['cn'].mean():.1f}")
    ax.axhline(p_cb, color=C3, linestyle=':', lw=1.2, label=f"Your baseline cb={p_cb}")
    ax.set_xlim(-1, 1)
    ax.set_xticks([0])
    ax.set_xticklabels(['Your Profile'])
    ax.set(ylabel='Predicted Creativity Now (cn)', title='P1 — Your Prediction + 95% CI')
    ax.legend()
    plt.tight_layout()
    show_plot(fig)

    with st.expander("📋 Full OLS Summary"):
        st.text(str(fit_p1.summary()))

# ══════════════════════════════════════════════
#  P2 — PREDICTION
# ══════════════════════════════════════════════
elif section == "🔮 P2 — Predict Dependency":
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">P2 — Are students more likely to be highly AI-dependent?</div>
      {badge("Lab 2 — Binomial MLE", "lab")} {badge("Lab 7 — Two-proportion Z-test", "lab")} {badge("INTERACTIVE PREDICTION", "pred")}
      <div class="need-box">
        <b>Research Need:</b> Use Binomial MLE to estimate the probability of high dependency per role,
        then test if students and professionals differ significantly.
      </div>
      <div class="var-grid">
        {pill('dep','cont')} <span class="pill-label"> Dependency index</span>
        &nbsp; {pill('role','cat')} <span class="pill-label"> Grouping variable</span>
        &nbsp; {pill('dep > threshold','target')} <span class="pill-label"> Binary high-dep classification</span>
      </div>
      {formula_box("L(p | k,n) = C(n,k)·pᵏ·(1-p)^(n-k)   →   MLE: p̂ = k/n")}
    </div>
    """, unsafe_allow_html=True)

    high_thresh = st.slider("High-dependency threshold (dep score)", 1.0, 5.0, 3.5, 0.1)

    stu_dep = df[df['role'] == 'Student']['dep']
    pro_dep = df[df['role'] == 'Working professional']['dep']
    ns, hs  = len(stu_dep), int((stu_dep > high_thresh).sum())
    np2, hp = len(pro_dep), int((pro_dep > high_thresh).sum())

    p_range    = np.linspace(0.001, 0.999, 1000)
    L_students = binom.pmf(hs, max(ns, 1), p_range)
    L_profs    = binom.pmf(hp, max(np2, 1), p_range)
    mle_s      = p_range[np.argmax(L_students)]
    mle_p      = p_range[np.argmax(L_profs)]

    se_s  = np.sqrt(mle_s * (1 - mle_s) / max(ns, 1))
    se_p  = np.sqrt(mle_p * (1 - mle_p) / max(np2, 1))
    ci_s  = [round(max(mle_s - 1.96 * se_s, 0), 3), round(min(mle_s + 1.96 * se_s, 1), 3)]
    ci_p  = [round(max(mle_p - 1.96 * se_p, 0), 3), round(min(mle_p + 1.96 * se_p, 1), 3)]

    z_p2, pval_p2 = 0.0, 1.0
    if ns > 0 and np2 > 0:
        try:
            # FIX: two-proportion Z-test comparing students vs professionals
            z_p2, pval_p2 = proportions_ztest([hs, hp], [ns, np2], alternative='larger')
        except Exception:
            pass

    risk_ratio = f"{mle_s / mle_p:.2f}x" if mle_p > 0.001 else "∞"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Students (n={ns})", f"{hs} high-dep", f"MLE p={mle_s:.3f}")
    c2.metric(f"Professionals (n={np2})", f"{hp} high-dep", f"MLE p={mle_p:.3f}")
    c3.metric("Risk Ratio", risk_ratio)
    c4.metric("Z-test p-value", f"{pval_p2:.4f}", delta="Significant" if pval_p2 < 0.05 else "Not sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(p_range, L_students, color=C2, lw=2, label=f'Students (MLE={mle_s:.3f})')
    axes[0].plot(p_range, L_profs,    color=C1, lw=2, label=f'Professionals (MLE={mle_p:.3f})')
    axes[0].axvline(mle_s, color=C2, linestyle='--', alpha=0.7)
    axes[0].axvline(mle_p, color=C1, linestyle='--', alpha=0.7)
    axes[0].set(xlabel='Probability p', ylabel='Binomial Likelihood',
                title='P2 — Binomial MLE Likelihood Curves')
    axes[0].legend()

    for i, (grp, phat, ci, c) in enumerate(zip(
            ['Students', 'Professionals'], [mle_s, mle_p], [ci_s, ci_p], [C2, C1])):
        axes[1].bar(grp, phat, color=c, alpha=0.75, width=0.4)
        axes[1].errorbar(i, phat, yerr=[[phat - ci[0]], [ci[1] - phat]],
                         fmt='none', color='white', capsize=8, lw=2)
        axes[1].text(i, phat + 0.02, f'{phat:.1%}', ha='center', fontsize=13,
                     fontweight='bold', color='white')
    axes[1].set(ylabel=f'Proportion with dep > {high_thresh}',
                title='P2 — High-Dependency Rates + 95% CI', ylim=(0, min(1.0, max(mle_s, mle_p) * 1.5 + 0.15)))
    plt.tight_layout()
    show_plot(fig)

    st.markdown("---")
    st.markdown("### 🎯 Interactive: Check Your Own Dependency")
    st.markdown("Enter your responses to estimate your dependency score and group:")

    col1, col2, col3 = st.columns(3)
    with col1:
        u_feel_less = st.slider("Feel less creative using AI (1=Strongly Disagree, 5=Strongly Agree)", 1, 5, 3)
    with col2:
        u_harder = st.slider("Independent thinking feels harder (1–5)", 1, 5, 3)
    with col3:
        u_accept = st.slider("Accept AI outputs without checking (1–10)", 1, 10, 5)

    # FIX: match dataset dep formula exactly: (feel_less + harder + accept/2) / 3
    u_dep  = (u_feel_less + u_harder + (u_accept / 2)) / 3
    u_high = u_dep > high_thresh
    u_role = st.radio("Your role:", ["Student", "Working professional"], horizontal=True)
    mle_ref = mle_s if u_role == "Student" else mle_p

    st.markdown(f"""
    <div style='background:#0e1e2e;border:1px solid #1e3a5f;border-radius:10px;padding:1.5rem;margin-top:1rem'>
      <div style='font-family:Space Mono,monospace;font-size:0.8rem;color:#5a7aaa'>YOUR DEPENDENCY SCORE</div>
      <div style='font-family:Space Mono,monospace;font-size:2.2rem;font-weight:700;color:#fff'>{u_dep:.2f} / 5.0</div>
      <div style='font-size:1rem;margin-top:0.5rem;color:{"#e8714a" if u_high else "#1D9E75"}'>
        {'⚠️ HIGH DEPENDENCY (above threshold)' if u_high else '✅ NORMAL DEPENDENCY (below threshold)'}
      </div>
      <div style='color:#5a7aaa;font-size:0.82rem;margin-top:0.4rem'>
        Among {u_role}s in our sample, {mle_ref:.1%} are classified as high-dependency.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if mle_p > 0.001:
        st.markdown(result_box(
            "✅ STUDENTS SIGNIFICANTLY MORE DEPENDENT" if pval_p2 < 0.05 else "❌ NOT SIGNIFICANT",
            f"z = {z_p2:.3f}, p = {pval_p2:.4f}. Risk ratio: {risk_ratio} (Students vs Professionals)."
        ), unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════
elif section == "📋 Summary":
    st.markdown("""
    <div class="section-card">
      <div class="section-title">📋 All Results — Final Summary</div>
      <div class="section-subtitle">Methods: Lab 1–11 · N respondents</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Re-run all models for summary ---
    t33, t67 = df['cb'].quantile(0.33), df['cb'].quantile(0.67)
    fit_a1   = smf.ols('change ~ cb', data=df).fit()
    fit_a2   = smf.ols('cn ~ hours + role_b + trained', data=df).fit()

    d_diff      = df['cn'] - df['cb']
    _, p_sw_sum = stats.shapiro(d_diff)
    t_pair_sum, p_pair_sum = stats.ttest_rel(df['cn'], df['cb'])
    pos_sum = int((df['change'] > 0).sum())
    neg_sum = int((df['change'] < 0).sum())
    total_nz = pos_sum + neg_sum
    if total_nz > 0:
        z_a3_sum, p_a3_sum = proportions_ztest(pos_sum, total_nz, value=0.5, alternative='larger')
    else:
        z_a3_sum, p_a3_sum = 0.0, 1.0

    feats4   = ['hours','writing','problem','creative','feel_less','harder','accept','conf']
    sc4_sum  = StandardScaler()
    X4_sum   = sc4_sum.fit_transform(df[feats4])
    y4_sum   = df['dep'].values
    lasso_sum = Lasso(alpha=0.05, max_iter=10000)
    lasso_sum.fit(X4_sum, y4_sum)
    kept_sum  = [f for f, c in zip(feats4, lasso_sum.coef_) if abs(c) > 1e-4]

    feats5   = ['hours','writing','problem','creative','trained']
    sc5_sum  = StandardScaler()
    X5_sum   = sc5_sum.fit_transform(df[feats5])
    y5_sum   = df['effort_sig'].values
    acc_sum  = 0.0
    if len(np.unique(y5_sum)) >= 2:
        lr5_sum = LogisticRegression(max_iter=1000, random_state=42)
        lr5_sum.fit(X5_sum, y5_sum)
        acc_sum = lr5_sum.score(X5_sum, y5_sum)

    fit_p1_sum = smf.ols('cn ~ cb + hours + creative + problem + conf + trained', data=df).fit()

    stu_dep_sum = df[df['role'] == 'Student']['dep']
    pro_dep_sum = df[df['role'] == 'Working professional']['dep']
    ns_sum  = len(stu_dep_sum);  hs_sum  = int((stu_dep_sum > 3.5).sum())
    np2_sum = len(pro_dep_sum);  hp_sum  = int((pro_dep_sum > 3.5).sum())
    p_rng   = np.linspace(0.001, 0.999, 1000)
    mle_s_sum = p_rng[np.argmax(binom.pmf(hs_sum, max(ns_sum, 1), p_rng))]
    mle_p_sum = p_rng[np.argmax(binom.pmf(hp_sum, max(np2_sum, 1), p_rng))]
    z_p2_sum, pval_p2_sum = 0.0, 1.0
    if ns_sum > 0 and np2_sum > 0:
        try:
            z_p2_sum, pval_p2_sum = proportions_ztest([hs_sum, hp_sum], [ns_sum, np2_sum], alternative='larger')
        except Exception:
            pass

    # FIX: use .iloc[0] instead of .get() for pandas Series access
    trained_pval = fit_a2.pvalues.iloc[3] if len(fit_a2.pvalues) > 3 else float('nan')
    trained_coef = fit_a2.params.iloc[3]  if len(fit_a2.params) > 3  else float('nan')

    summary_rows = [
        ("A1", "ANOVA + OLS Regression",     "Lab 8+10",
         f"slope={fit_a1.params['cb']:.3f}, p={fit_a1.pvalues['cb']:.4f}",
         "Significant" if fit_a1.pvalues['cb'] < 0.05 else "Not sig"),
        ("A2", "Multiple OLS (dummies)",      "Lab 10",
         f"R²={fit_a2.rsquared:.3f}, trained coef={trained_coef:.3f}",
         "Significant" if (not np.isnan(trained_pval) and trained_pval < 0.05) else "Not sig"),
        ("A3", "Paired t + Shapiro + 2-prop Z", "Lab 9+7",
         f"{pos_sum} improved, {neg_sum} declined. z={z_a3_sum:.3f}, p={p_a3_sum:.4f}",
         "Significant" if p_a3_sum < 0.05 else "Not sig"),
        ("A4", "Lasso Regression",            "Lab 10",
         f"Kept: {kept_sum}. R²={lasso_sum.score(X4_sum,y4_sum):.3f}",
         "Variable selection"),
        ("A5", "Logistic Regression",         "Lab 11",
         f"Accuracy={acc_sum*100:.0f}%",
         "Classification"),
        ("A6", "Two-sample t-test",           "Lab 6",
         "Students vs Professionals confidence gain from training",
         "Exploratory"),
        ("A7", "Two-Way ANOVA + Tukey",       "Lab 8",
         "Usage type × role interaction tested",
         "ANOVA"),
        ("P1", "OLS Prediction + 95% CI",     "Lab 10",
         f"R²={fit_p1_sum.rsquared:.3f}, RMSE={np.sqrt(fit_p1_sum.mse_resid):.3f}",
         "Predictive"),
        ("P2", "Binomial MLE + 2-prop Z",     "Lab 2+7",
         f"Students {mle_s_sum:.1%} vs Profs {mle_p_sum:.1%}. z={z_p2_sum:.3f}, p={pval_p2_sum:.4f}",
         "Significant" if pval_p2_sum < 0.05 else "Not sig"),
    ]

    sdf = pd.DataFrame(summary_rows, columns=["Code", "Method", "Lab Ref", "Key Finding", "Result"])
    st.dataframe(sdf, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div style='display:flex;gap:1rem;flex-wrap:wrap;margin-top:1.5rem'>
      <div class='metric-card'><div class='metric-val'>{n}</div><div class='metric-lbl'>Respondents</div></div>
      <div class='metric-card'><div class='metric-val'>9</div><div class='metric-lbl'>Analyses</div></div>
      <div class='metric-card'><div class='metric-val'>Lab 2–11</div><div class='metric-lbl'>Methods Used</div></div>
      <div class='metric-card'><div class='metric-val'>{pos_sum}/{pos_sum+neg_sum}</div><div class='metric-lbl'>Improved</div></div>
      <div class='metric-card'><div class='metric-val'>{fit_p1_sum.rsquared:.2f}</div><div class='metric-lbl'>Best R²</div></div>
    </div>
    """, unsafe_allow_html=True)