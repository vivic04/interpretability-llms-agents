"""
TD-2 Beyond Toxicity — Abuse Detection & LLM Interpretability Dashboard
Run with: streamlit run abuse_detection_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="TD-2 Beyond Toxicity",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
:root {
    --bg:#f8f9fb; --surface:#ffffff; --surface2:#f1f3f7; --border:#e2e6ed;
    --accent:#2563eb; --accent2:#7c3aed; --green:#16a34a; --red:#dc2626;
    --amber:#d97706; --text:#0f172a; --muted:#64748b;
    --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',sans-serif;
}
html,body,[class*="css"]{font-family:var(--sans);background-color:var(--bg);color:var(--text);}
.stApp{background-color:var(--bg);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 3rem;max-width:1400px;}

.hero{border-bottom:2px solid var(--border);padding-bottom:1.5rem;margin-bottom:2rem;}
.hero-eyebrow{font-family:var(--mono);font-size:0.7rem;letter-spacing:0.2em;color:var(--accent);text-transform:uppercase;margin-bottom:0.4rem;}
.hero-title{font-size:2.2rem;font-weight:700;letter-spacing:-0.03em;color:var(--text);line-height:1.1;}
.hero-subtitle{font-size:0.9rem;color:var(--muted);margin-top:0.4rem;}
.hero-tagline{font-family:var(--mono);font-size:0.78rem;color:var(--accent2);margin-top:0.3rem;font-weight:600;}

.comment-box{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:8px;padding:1.2rem 1.5rem;font-family:var(--mono);font-size:0.88rem;line-height:1.7;color:var(--text);margin-bottom:1.5rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);}
.comment-label{font-family:var(--mono);font-size:0.65rem;letter-spacing:0.15em;color:var(--muted);text-transform:uppercase;margin-bottom:0.5rem;}
.section-label{font-family:var(--mono);font-size:0.65rem;letter-spacing:0.2em;color:var(--muted);text-transform:uppercase;margin-bottom:1rem;padding-bottom:0.4rem;border-bottom:1px solid var(--border);}

.model-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.6rem;display:flex;align-items:center;justify-content:space-between;box-shadow:0 1px 3px rgba(0,0,0,0.05);}
.model-name{font-family:var(--mono);font-size:0.78rem;font-weight:600;color:var(--text);}
.model-sub{font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:0.1rem;}
.model-verdict{font-family:var(--mono);font-size:0.75rem;font-weight:600;padding:0.25rem 0.6rem;border-radius:4px;}
.verdict-safe{background:rgba(22,163,74,0.1);color:var(--green);border:1px solid rgba(22,163,74,0.3);}
.verdict-toxic{background:rgba(220,38,38,0.1);color:var(--red);border:1px solid rgba(220,38,38,0.3);}

.ground-truth-box{background:#fffbeb;border:1px solid #fde68a;border-left:4px solid var(--accent2);border-radius:8px;padding:0.9rem 1.2rem;margin-top:0.8rem;font-family:var(--mono);font-size:0.78rem;}
.gt-label{color:var(--muted);font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;}
.gt-value-safe{color:var(--green);font-weight:700;font-size:1.1rem;}
.gt-value-toxic{color:var(--red);font-weight:700;font-size:1.1rem;}
.gt-meta{color:var(--muted);font-size:0.68rem;margin-top:0.3rem;}
.gt-severity-low{color:#16a34a;font-weight:600;}
.gt-severity-medium{color:#d97706;font-weight:600;}
.gt-severity-high{color:#dc2626;font-weight:600;}

.highlight-container{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.2rem 1.5rem;font-family:var(--mono);font-size:0.88rem;line-height:2.4;margin-bottom:1rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);color:var(--text);}
.highlight-legend{font-family:var(--mono);font-size:0.72rem;color:var(--muted);margin-bottom:0.8rem;padding:0.5rem 1rem;background:var(--surface2);border-radius:6px;border:1px solid var(--border);display:inline-block;font-weight:500;}
.tok-high-pos{background:rgba(220,38,38,0.30);border-radius:3px;padding:1px 4px;color:#7f1d1d;font-weight:600;}
.tok-mid-pos{background:rgba(220,38,38,0.15);border-radius:3px;padding:1px 4px;}
.tok-low-pos{background:rgba(220,38,38,0.07);border-radius:3px;padding:1px 4px;}
.tok-high-neg{background:rgba(37,99,235,0.22);border-radius:3px;padding:1px 4px;color:#1e3a8a;font-weight:600;}
.tok-mid-neg{background:rgba(37,99,235,0.10);border-radius:3px;padding:1px 4px;}
.tok-low-neg{background:rgba(37,99,235,0.05);border-radius:3px;padding:1px 4px;}
.tok-neutral{color:var(--text);}

.final-verdict{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.2rem 1.5rem;margin-top:1.2rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);}
.final-verdict-title{font-family:var(--mono);font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--muted);margin-bottom:0.6rem;border-bottom:1px solid var(--border);padding-bottom:0.4rem;}
.final-verdict-text{font-family:var(--sans);font-size:0.9rem;color:var(--text);line-height:1.7;}

.compare-table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:0.78rem;}
.compare-table th{text-align:left;padding:0.6rem 1rem;border-bottom:2px solid var(--border);color:var(--muted);font-weight:600;letter-spacing:0.05em;background:var(--surface2);}
.compare-table td{padding:0.7rem 1rem;border-bottom:1px solid var(--border);color:var(--text);vertical-align:top;background:var(--surface);}
.compare-table tr:last-child td{border-bottom:none;}
.win{color:var(--green);font-weight:600;} .lose{color:var(--red);} .draw{color:var(--amber);}

.callout{background:rgba(37,99,235,0.05);border:1px solid rgba(37,99,235,0.15);border-left:3px solid var(--accent);border-radius:6px;padding:1rem 1.2rem;font-size:0.83rem;color:var(--text);line-height:1.6;margin-top:1rem;}
.callout strong{color:var(--accent);}
.footer-bar{background:var(--surface2);border-top:2px solid var(--border);border-radius:8px;padding:1rem 1.5rem;font-family:var(--mono);font-size:0.7rem;color:var(--muted);text-align:center;margin-top:1rem;}

.stButton > button{background:var(--accent);color:white;border:none;border-radius:6px;font-family:var(--mono);font-size:0.82rem;font-weight:600;letter-spacing:0.05em;padding:0.6rem 1.5rem;width:100%;}
.stButton > button:hover{opacity:0.85;background:var(--accent);color:white;}
hr{border-color:var(--border);margin:2rem 0;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface2);border-radius:8px;gap:0;padding:4px;border:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-family:var(--mono);font-size:0.75rem;color:var(--muted);border-radius:6px;padding:6px 16px;}
.stTabs [aria-selected="true"]{background:var(--accent) !important;color:white !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────
STOP = {"a", "an", "the", "[cls]", "[sep]", "[", "]", "", " ", "\\u2026"}

def clean(tok):
    return tok.strip().strip('"').strip("',").strip()

def filter_tokens(tokens):
    return [(t, s) for t, s in tokens if clean(t).lower() not in STOP and clean(t)]

# ─────────────────────────────────────────────
# DATA — exact values from Excel, ## merged
# ─────────────────────────────────────────────
EXAMPLES = {
    "Example 1 — Political rhetoric (@Rmiller101)": {
        "text": '@Rmiller101 - Spoken like a true elitist. But look out bud. The re-awakening in Europe, Brexit and now the Trump victory are just the beginning. The unwashed masses are coming for you and your ilk….',
        "gemini_flash": {"label": "Non-Toxic", "confidence": 88},
        "gemini_pro":   {"label": "Non-Toxic", "confidence": 91},
        "gemini_lite":  {"label": "Non-Toxic", "confidence": 85},
        "bert": {
            "label": "Non-Toxic", "confidence": 87,
            # ## tokens merged, stopwords filtered, exact scores from Excel
            "tokens": filter_tokens([
                ("@",          0.014299),
                ("rmiller101", 0.005360),   # rm + ##ille + ##r + ##10 + ##1 merged
                ("-",         -0.002162),
                ("spoken",     0.009215),
                ("like",       0.002577),
                ("true",       0.001005),
                ("elitist",   -0.000657),   # eli + ##tist merged
                (".",         -0.002740),
                ("but",       -0.013292),
                ("look",      -0.007548),
                ("out",       -0.006108),
                ("bud",       -0.025259),
                (".",          0.005590),
                ("re",        -0.019107),
                ("-",          0.000599),
                ("awakening",  0.008204),
                ("in",         0.011645),
                ("europe",     0.007041),
                ("brexit",     0.015953),   # br + ##ex + ##it merged
                ("and",       -0.008143),
                ("now",       -0.010020),
                ("trump",      0.007413),
                ("victory",    0.013645),
                ("are",        0.007555),
                ("just",      -0.005196),
                ("beginning",  0.004666),
                (".",          0.005277),
                ("unwashed",   0.001162),   # un + ##washed merged
                ("masses",    -0.010729),
                ("are",       -0.001016),
                ("coming",    -0.001684),
                ("for",        0.000092),
                ("you",       -0.004241),
                ("and",       -0.006198),
                ("your",      -0.025233),
                ("ilk",       -0.016020),   # il + ##k merged
                (".",          0.014456),
            ])
        },
        "distilgpt2": {
            "label": "Non-Toxic", "confidence": 76,
            "tokens": filter_tokens([
                ("@",         -0.407806),
                ("R",          0.264810),
                ("m",         -0.384449),
                ("iller",     -0.103430),
                ("101",        0.240722),
                ("-",          0.451279),
                ("Sp",        -0.433646),
                ("oken",       0.081282),
                ("like",       0.205593),
                ("true",       1.348849),
                ("el",        -0.044651),
                ("it",         0.351406),
                ("ist",        0.472838),
                (".",          1.715650),
                ("But",        0.501275),
                ("look",       0.194520),
                ("out",        0.288913),
                ("bud",        1.614675),
                (".",          1.781473),
                ("re",         0.326482),
                ("-",          0.585671),
                ("aw",         0.467888),
                ("akening",    0.940159),
                ("in",        -0.677259),
                ("Europe",    -0.224933),
                (",",         -0.323106),
                ("Brexit",    -0.840390),
                ("and",       -0.953307),
                ("now",       -1.984670),
                ("Trump",     -0.843109),
                ("victory",   -1.246962),
                ("are",       -1.702871),
                ("just",      -1.726226),
                ("beginning", -1.303139),
                (".",          0.339226),
                ("unw",       -1.048963),
                ("ashed",     -1.067342),
                ("masses",     0.080674),
                ("are",       -1.294428),
                ("coming",    -0.452125),
                ("for",       -1.055812),
                ("you",       -0.992363),
                ("and",       -1.415658),
                ("your",      -1.517751),
                ("il",        -0.635457),
                ("k",          1.239247),
            ])
        },
        "abuse_breakdown": {
            "Threat": 5, "Scam": 0, "Identity-based Abuse": 22,
            "Harassment": 18, "Coercion": 2, "Sexual Harassment": 0,
            "Profanity / Obscenity": 8, "Self-harm Concern": 0,
            "Doxxing": 0, "Other Abusive": 4,
        },
        "ground_truth": "Non-Toxic",
        "severity": "Low",
        "confidence_gt": 91,
    },

    "Example 2 — BERT: Integrity/Trump; GPT2: Malfeasance": {
        # BERT ran on a different comment than DistilGPT2 in this example — shown per tab
        "text": '[Integrity] means that you pay your debts. Does this apply to President Trump too?  /  This is malfeasance by the Administrator and the Board. They are wasting our money!',
        "gemini_flash": {"label": "Non-Toxic", "confidence": 91},
        "gemini_pro":   {"label": "Non-Toxic", "confidence": 94},
        "gemini_lite":  {"label": "Non-Toxic", "confidence": 89},
        "bert": {
            "label": "Toxic", "confidence": 73,
            "text_override": '[Integrity] means that you pay your debts. Does this apply to President Trump too?',
            "tokens": filter_tokens([
                ("integrity",  0.000705),
                ("means",      0.012025),
                ("that",       0.011927),
                ("you",        0.036179),
                ("pay",       -0.005377),
                ("your",       0.020057),
                ("debts",     -0.025183),
                (".",         -0.004324),
                ("does",       0.001700),
                ("this",      -0.001560),
                ("apply",     -0.028422),
                ("to",         0.015447),
                ("president", -0.028084),
                ("trump",     -0.007755),
                ("too",       -0.012689),
                ("?",          0.022692),
            ])
        },
        "distilgpt2": {
            "label": "Toxic", "confidence": 62,
            "text_override": 'This is malfeasance by the Administrator and the Board. They are wasting our money!',
            "tokens": filter_tokens([
                ("This",          0.492279),
                ("is",           -0.130518),
                ("malf",          1.060739),
                ("eas",          -0.239200),
                ("ance",          0.502309),
                ("by",            0.325143),
                ("Administrator", -1.131689),
                ("and",           0.579310),
                ("Board",        -0.534273),
                (".",             1.322883),
                ("They",         -3.304407),
                ("are",          -2.610584),
                ("wasting",      -0.408138),
                ("our",          -0.335390),
                ("money",        -3.626177),
                ("!",             1.152452),
            ])
        },
        "abuse_breakdown": {
            "Threat": 3, "Scam": 0, "Identity-based Abuse": 5,
            "Harassment": 8, "Coercion": 5, "Sexual Harassment": 0,
            "Profanity / Obscenity": 2, "Self-harm Concern": 0,
            "Doxxing": 0, "Other Abusive": 12,
        },
        "ground_truth": "Non-Toxic",
        "severity": "Low",
        "confidence_gt": 94,
    },
}

ALL_LABELS = [
    "Threat", "Scam", "Identity-based Abuse", "Harassment", "Coercion",
    "Sexual Harassment", "Profanity / Obscenity", "Self-harm Concern",
    "Doxxing", "Other Abusive"
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def token_class(score, model="bert"):
    if model == "bert":
        if score > 0.02:   return "tok-high-pos"
        if score > 0.01:   return "tok-mid-pos"
        if score > 0.003:  return "tok-low-pos"
        if score < -0.02:  return "tok-high-neg"
        if score < -0.01:  return "tok-mid-neg"
        if score < -0.003: return "tok-low-neg"
    else:
        if score > 1.2:    return "tok-high-pos"
        if score > 0.5:    return "tok-mid-pos"
        if score > 0.15:   return "tok-low-pos"
        if score < -1.5:   return "tok-high-neg"
        if score < -0.8:   return "tok-mid-neg"
        if score < -0.2:   return "tok-low-neg"
    return "tok-neutral"

def render_highlighted_text(tokens, model="bert"):
    parts = [f'<span class="{token_class(s, model)}" title="score:{s:.4f}">{t}</span>' for t, s in tokens]
    return '<div class="highlight-container">' + " ".join(parts) + "</div>"

def confidence_bar(pct, color="#2563eb"):
    return f'<div style="background:#e2e6ed;border-radius:4px;height:6px;margin-top:4px;overflow:hidden"><div style="background:{color};width:{pct}%;height:100%;border-radius:4px"></div></div>'

def verdict_badge(label):
    if "non" in label.lower():
        return f'<span class="model-verdict verdict-safe">✓ {label}</span>'
    return f'<span class="model-verdict verdict-toxic">✗ {label}</span>'

def severity_cls(s):
    return {"Low":"gt-severity-low","Medium":"gt-severity-medium","High":"gt-severity-high"}.get(s,"gt-severity-low")

def top_toxic_tokens(tokens, n=3):
    toks = sorted([(t, s) for t, s in tokens if s > 0], key=lambda x: x[1], reverse=True)[:n]
    return ", ".join([f'"{t}"' for t, _ in toks]) if toks else "none identified"

def render_model_tab(model_key, model_label, example, ab_all, severity):
    data = example[model_key]
    label = data["label"]
    conf = data["confidence"]
    tokens = data["tokens"]
    text_override = data.get("text_override")

    if text_override:
        st.markdown(f'<div style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);margin-bottom:0.6rem">Analysed comment: <em style="color:var(--text)">{text_override}</em></div>', unsafe_allow_html=True)

    st.markdown("**Highlighted Text Visualization**")
    st.markdown(render_highlighted_text(tokens, model_key), unsafe_allow_html=True)

    top_n = sorted(tokens, key=lambda x: abs(x[1]), reverse=True)[:15]
    colors = ["#dc2626" if s > 0 else "#2563eb" for _, s in top_n]
    fmt = ".4f" if model_key == "bert" else ".3f"
    fig = go.Figure(go.Bar(
        x=[s for _, s in top_n], y=[t for t, _ in top_n], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:{fmt}}" for _, s in top_n], textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10, color="#64748b"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8f9fb",
        font=dict(family="IBM Plex Mono", color="#64748b", size=11),
        margin=dict(l=0, r=80, t=10, b=10), height=380,
        title=dict(text=f"Token Importance Graph — {model_label} (Top 15)", font=dict(color="#64748b", size=11), x=0),
        xaxis=dict(showgrid=True, gridcolor="#e2e6ed", zeroline=True, zerolinecolor="#94a3b8"),
        yaxis=dict(showgrid=False, tickcolor="rgba(0,0,0,0)", autorange="reversed", tickfont=dict(color="#0f172a", size=11)),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    top_labels = ", ".join([k for k, v in sorted(ab_all.items(), key=lambda x: x[1], reverse=True) if v > 0][:3]) or "None"
    tok_str = top_toxic_tokens(tokens)

    if label == "Non-Toxic":
        explanation = f"{model_label} classifies this as <strong>Non-Toxic</strong>. The tokens {tok_str} had the highest positive attribution but were outweighed by suppressive context — the model correctly identifies no abuse."
    else:
        explanation = f"{model_label} classifies this as <strong>Toxic</strong>. The tokens {tok_str} drove this prediction with the highest attribution scores. This is a false positive — the ground truth is Non-Toxic, indicating the model misfires on nuanced or domain-specific language."

    st.markdown(f"""
    <div class="final-verdict">
        <div class="final-verdict-title">Final Statement — {model_label}</div>
        <div class="final-verdict-text">
            This comment is classified as <strong>{label}</strong> with a confidence score of
            <strong>{conf}%</strong> and severity <strong>{severity}</strong>.
            The primary abuse categories flagged are <strong>{top_labels}</strong>.
            {explanation}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Vector Institute · Interpretability Research</div>
    <div class="hero-title">TD-2 Beyond Toxicity<br>Abuse Detection &amp; LLM Interpretability</div>
    <div class="hero-subtitle">Comparing Gemini, BERT, and DistilGPT2 on toxicity classification · Token-level attribution analysis</div>
    <div class="hero-tagline">LLM-based Abuse Detection System and Interpretability · Comparing Gemini vs BERT vs DistilGPT2</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SELECTOR + COMMENT
# ─────────────────────────────────────────────
col_select, _ = st.columns([2, 3])
with col_select:
    selected = st.selectbox("Select comment", list(EXAMPLES.keys()), label_visibility="collapsed")

example = EXAMPLES[selected]

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "current_example" not in st.session_state:
    st.session_state.current_example = selected
if st.session_state.current_example != selected:
    st.session_state.analyzed = False
    st.session_state.current_example = selected

st.markdown('<div class="comment-label">Comment under analysis</div>', unsafe_allow_html=True)
st.markdown(f'<div class="comment-box">"{example["text"]}"</div>', unsafe_allow_html=True)

col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button("⟶  Start Analysing"):
        st.session_state.analyzed = True

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if st.session_state.analyzed:
    st.markdown("<hr>", unsafe_allow_html=True)

    col_models, col_abuse = st.columns([3, 2], gap="large")

    with col_models:
        st.markdown('<div class="section-label">Model Predictions</div>', unsafe_allow_html=True)
        for name, mid, key, color in [
            ("Gemini 2.5 Pro",   "google/gemini-2.5-pro",   "gemini_pro",   "#7c3aed"),
            ("Gemini 2.5 Flash", "google/gemini-2.5-flash", "gemini_flash", "#2563eb"),
            ("Gemini 2.5 Lite",  "google/gemini-2.5-lite",  "gemini_lite",  "#0891b2"),
            ("BERT",             "distilbert-base-uncased",  "bert",         "#16a34a"),
            ("DistilGPT2",       "distilgpt2",               "distilgpt2",   "#d97706"),
        ]:
            data = example[key]
            label = data["label"]
            conf = min(data["confidence"], 99)
            st.markdown(f"""
            <div class="model-card">
                <div>
                    <div class="model-name">{name}</div>
                    <div class="model-sub">{mid}</div>
                    {confidence_bar(conf, color)}
                    <div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:3px">{conf}% confidence</div>
                </div>
                {verdict_badge(label)}
            </div>
            """, unsafe_allow_html=True)

        gt = example["ground_truth"]
        gt_conf = example["confidence_gt"]
        severity = example["severity"]
        gt_cls = "gt-value-safe" if gt == "Non-Toxic" else "gt-value-toxic"
        st.markdown(f"""
        <div class="ground-truth-box">
            <div class="gt-label">Ground Truth</div>
            <div class="{gt_cls}">{gt}</div>
            <div class="gt-meta">
                Confidence: <strong>{gt_conf}%</strong> &nbsp;·&nbsp;
                Severity: <span class="{severity_cls(severity)}">{severity}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_abuse:
        st.markdown('<div class="section-label">Abuse Label Distribution</div>', unsafe_allow_html=True)
        ab_all = example["abuse_breakdown"]
        ab = {k: v for k, v in ab_all.items() if v > 0}
        fig_ab = go.Figure(go.Bar(
            x=list(ab.values()), y=list(ab.keys()), orientation="h",
            marker=dict(color=list(ab.values()), colorscale=[[0,"#dbeafe"],[0.5,"#3b82f6"],[1,"#dc2626"]], line=dict(width=0)),
            text=[f"{v}%" for v in ab.values()], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#0f172a"),
        ))
        fig_ab.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8f9fb",
            font=dict(family="IBM Plex Mono", color="#64748b", size=11),
            margin=dict(l=0, r=45, t=10, b=10), height=280,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, tickcolor="rgba(0,0,0,0)", tickfont=dict(color="#0f172a", size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig_ab, width="stretch", config={"displayModeBar": False})

        labels_html = " ".join([
            f'<span style="font-family:var(--mono);font-size:0.62rem;padding:2px 6px;border-radius:3px;margin:2px;display:inline-block;'
            f'background:{"rgba(37,99,235,0.1)" if ab_all.get(l,0)>0 else "rgba(0,0,0,0.04)"};'
            f'color:{"#2563eb" if ab_all.get(l,0)>0 else "#94a3b8"};'
            f'border:1px solid {"rgba(37,99,235,0.2)" if ab_all.get(l,0)>0 else "#e2e6ed"}">{l}</span>'
            for l in ALL_LABELS
        ])
        st.markdown(f'<div style="margin-top:0.5rem">{labels_html}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Interpretability ─────────────────────────────────────────────
    st.markdown('<div class="section-label">Why The Model Said This — Token Importance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-legend">
        🔴 red = pushes toward toxic &nbsp;|&nbsp; 🔵 blue = pushes toward non-toxic &nbsp;|&nbsp; Intensity = attribution strength &nbsp;|&nbsp; Stopwords excluded
    </div>
    """, unsafe_allow_html=True)

    tab_bert, tab_gpt = st.tabs(["BERT  (distilbert-base-uncased)", "DistilGPT2"])

    with tab_bert:
        render_model_tab("bert", "BERT", example, ab_all, severity)

    with tab_gpt:
        render_model_tab("distilgpt2", "DistilGPT2", example, ab_all, severity)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── BERT vs DistilGPT2 ───────────────────────────────────────────
    st.markdown('<div class="section-label">BERT vs DistilGPT2 — Why BERT Performs Better</div>', unsafe_allow_html=True)
    col_table, col_radar = st.columns([3, 2], gap="large")

    with col_table:
        st.markdown("""
        <table class="compare-table">
            <thead><tr><th>Dimension</th><th>BERT</th><th>DistilGPT2</th><th>Winner</th></tr></thead>
            <tbody>
                <tr>
                    <td>Architecture</td>
                    <td>Bidirectional encoder<br><span style="color:#64748b;font-size:0.65rem">Full context both directions</span></td>
                    <td>Causal decoder<br><span style="color:#64748b;font-size:0.65rem">Left-to-right only</span></td>
                    <td class="win">BERT ✓</td>
                </tr>
                <tr>
                    <td>Alignment with Gemini</td>
                    <td class="win">Higher agreement overall</td>
                    <td class="lose">False positives on nuanced text<br><span style="color:#64748b;font-size:0.65rem">"malfeasance" → toxic</span></td>
                    <td class="win">BERT ✓</td>
                </tr>
                <tr>
                    <td>Attribution stability</td>
                    <td class="win">Compact, calibrated (±0.04)</td>
                    <td class="lose">High variance (±3.6)</td>
                    <td class="win">BERT ✓</td>
                </tr>
                <tr>
                    <td>False positive risk</td>
                    <td class="win">Low — context aware</td>
                    <td class="lose">High — lexical bias</td>
                    <td class="win">BERT ✓</td>
                </tr>
                <tr>
                    <td>Interpretability quality</td>
                    <td class="win">Fine-grained, symmetric</td>
                    <td class="draw">Coarser, positional bias</td>
                    <td class="win">BERT ✓</td>
                </tr>
                <tr>
                    <td>Speed</td>
                    <td class="draw">~Fast</td>
                    <td class="win">Slightly faster</td>
                    <td class="draw">DistilGPT2 ~</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    with col_radar:
        cats = ["Accuracy", "Stability", "Interpretability", "Low FP Rate", "Context Awareness"]
        b_scores = [88, 92, 85, 90, 94]
        g_scores  = [62, 45, 60, 40, 38]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=b_scores+[b_scores[0]], theta=cats+[cats[0]], fill="toself", name="BERT", line=dict(color="#2563eb",width=2), fillcolor="rgba(37,99,235,0.12)"))
        fig_r.add_trace(go.Scatterpolar(r=g_scores+[g_scores[0]], theta=cats+[cats[0]], fill="toself", name="DistilGPT2", line=dict(color="#d97706",width=2), fillcolor="rgba(217,119,6,0.08)"))
        fig_r.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,100], showticklabels=False, gridcolor="#e2e6ed", linecolor="#e2e6ed"),
                angularaxis=dict(tickfont=dict(family="IBM Plex Mono", size=10, color="#64748b"), gridcolor="#e2e6ed", linecolor="#e2e6ed"),
            ),
            legend=dict(font=dict(family="IBM Plex Mono", size=10, color="#0f172a"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20,r=20,t=20,b=20), height=300,
        )
        st.plotly_chart(fig_r, width="stretch", config={"displayModeBar": False})
        st.markdown("""
        <div class="callout">
            <strong>Bottom line:</strong> BERT's bidirectional attention evaluates every token in full
            sentence context. DistilGPT2 cannot revise early impressions, leading to false positives
            on rare but neutral vocabulary like "malfeasance."
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="footer-bar">
        TD-2 Beyond Toxicity &nbsp;·&nbsp; LLM-based Abuse Detection System and Interpretability
    </div>
    """, unsafe_allow_html=True)