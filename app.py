import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import os
import shap
import matplotlib.pyplot as plt

# ============================================================
# model aur config load kar rahe hai
# ============================================================
model = joblib.load('fraud_model.pkl')
with open('model_config.json', 'r') as f:
    config = json.load(f)
threshold = config['threshold']
feature_names = config['features']

@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)

explainer = get_explainer()

# agar demo_examples.json bani hui hai to load kar lo (real fraud/legit rows)
demo_examples = None
if os.path.exists('demo_examples.json'):
    with open('demo_examples.json', 'r') as f:
        demo_examples = json.load(f)

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# thoda CSS daal diya taki UI achi lage, dark theme wala dashboard
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #131722 0%, #0b0d12 55%, #08090c 100%);
    }

    /* Hero header */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6ee7ff 0%, #7c8cff 45%, #b98cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #8b93a7;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #161a24, #10131b);
        border: 1px solid #232838;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }
    div[data-testid="stMetricLabel"] { color: #8b93a7 !important; }
    div[data-testid="stMetricValue"] {
        color: #e7ecf7 !important;
        font-weight: 700;
    }

    /* Section headers */
    h2, h3 {
        color: #e7ecf7 !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #8b93a7 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(145deg, #1b2030, #12151e);
        color: #e7ecf7;
        border: 1px solid #2b3145;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.15s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .stButton > button:hover {
        border-color: #6ee7ff;
        color: #6ee7ff;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(110,231,255,0.15);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Primary "Detect Fraud" button — target via key wrapper */
    div[data-testid="stButton"] button[kind="secondaryFormSubmit"],
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #ff5c7c 0%, #ff8a5c 100%);
        border: none;
        color: white;
        font-size: 1.05rem;
        padding: 0.7rem 0;
    }

    /* Inputs */
    input, textarea, select,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {
        background-color: #12151e !important;
        border-radius: 8px !important;
        border: 1px solid #232838 !important;
        color: #e7ecf7 !important;
    }
    label, .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #b7bfd1 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    /* Alert / result boxes */
    div[data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 1rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    }

    /* Divider */
    hr { border-color: #1d2230 !important; }

    /* Slider track */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #6ee7ff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">💳 Advanced Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-powered real-time fraud detection · IEEE-CIS Dataset · Random Forest + SHAP Explainability</div>', unsafe_allow_html=True)
st.divider()

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Model", "Random Forest")
mcol2.metric("AUC Score", "0.9212")
mcol3.metric("Fraud Recall", "62%")
mcol4.metric("Total Transactions", "5,90,540")
st.divider()

# ============================================================
# encoding wale maps - ye category ko number me convert karne ke liye hai
# ============================================================
card4_map = {"visa": 2, "mastercard": 3, "discover": 1, "american express": 0}
card6_map = {"debit": 0, "credit": 1, "debit or credit": 2, "charge card": 3}
product_map = {"W": 4, "H": 2, "C": 1, "S": 3, "R": 2}
device_map = {"desktop": 0, "mobile": 1, "unknown": 2}
email_map = {"gmail.com": 17, "yahoo.com": 55, "hotmail.com": 24, "outlook.com": 41,
             "icloud.com": 20, "anonymous.com": 2, "aol.com": 3, "comcast.net": 8, "other": 0}

# ulta map - jab preset load karna ho to number se wapas naam pe aana padta hai
rev_card4 = {v: k for k, v in card4_map.items()}
rev_card6 = {v: k for k, v in card6_map.items()}
rev_product = {v: k for k, v in product_map.items()}
rev_device = {v: k for k, v in device_map.items()}
rev_email = {v: k for k, v in email_map.items()}

WIDGET_KEYS = ["transaction_amt", "card1", "card_type", "card_category", "email_domain",
               "addr1", "c1", "c2", "c4", "c6", "c8", "c10", "c11", "c13", "c14",
               "hour", "product", "device_type",
               "v29", "v51", "v70", "v91", "v279", "v280", "v302", "v303", "v304"]


def load_preset(ex):
    """Fill the manual-entry widgets AND remember the full 223-feature row
    so Detect Fraud uses the exact real row (overlaid with any manual edits)."""
    feat = ex["features"]
    st.session_state["preset_full"] = feat  # full real row, used as the base at predict time

    st.session_state["transaction_amt"] = float(feat.get("TransactionAmt", 100.0))
    st.session_state["card1"] = int(feat.get("card1", 9000))
    st.session_state["card_type"] = rev_card4.get(int(feat.get("card4", 2)), "visa")
    st.session_state["card_category"] = rev_card6.get(int(feat.get("card6", 0)), "debit")
    st.session_state["email_domain"] = rev_email.get(int(feat.get("P_emaildomain", 17)), "gmail.com")
    st.session_state["addr1"] = float(feat.get("addr1", 100.0))
    st.session_state["c1"] = int(feat.get("C1", 1))
    st.session_state["c2"] = int(feat.get("C2", 1))
    st.session_state["c4"] = int(feat.get("C4", 0))
    st.session_state["c6"] = int(feat.get("C6", 1))
    st.session_state["c8"] = int(feat.get("C8", 1))
    st.session_state["c10"] = int(feat.get("C10", 1))
    st.session_state["c11"] = int(feat.get("C11", 1))
    st.session_state["c13"] = int(feat.get("C13", 3))
    st.session_state["c14"] = int(feat.get("C14", 1))
    st.session_state["hour"] = int(feat.get("Transaction_Hour", 12))
    st.session_state["product"] = rev_product.get(int(feat.get("ProductCD", 4)), "W")
    st.session_state["device_type"] = rev_device.get(int(feat.get("DeviceType", 0)), "desktop")
    st.session_state["v29"] = float(feat.get("V29", 0.0))
    st.session_state["v51"] = float(feat.get("V51", 0.0))
    st.session_state["v70"] = float(feat.get("V70", 0.0))
    st.session_state["v91"] = float(feat.get("V91", 0.0))
    st.session_state["v279"] = float(feat.get("V279", 0.0))
    st.session_state["v280"] = float(feat.get("V280", 0.0))
    st.session_state["v302"] = float(feat.get("V302", 0.0))
    st.session_state["v303"] = float(feat.get("V303", 0.0))
    st.session_state["v304"] = float(feat.get("V304", 0.0))


# form render hone se pehle ek baar default values set kar do session_state me
# (agar already nahi hai to) - naye streamlit me key+value dono dena allowed nahi hai
_defaults = {
    "transaction_amt": 100.0, "card1": 9000, "card_type": "visa", "card_category": "debit",
    "email_domain": "gmail.com", "addr1": 100.0, "c1": 1, "c2": 1, "c4": 0, "c6": 1, "c8": 1,
    "c10": 1, "c11": 1, "c13": 3, "c14": 1, "hour": 12, "product": "W", "device_type": "desktop",
    "v29": 0.0, "v51": 0.0, "v70": 0.0, "v91": 0.0, "v279": 0.0, "v280": 0.0,
    "v302": 0.0, "v303": 0.0, "v304": 0.0,
}
for _k, _v in _defaults.items():
    st.session_state.setdefault(_k, _v)

# ============================================================
# ye demo wale buttons hai - click karte hi form niche wala fill ho jayega
# real data se, phir Detect Fraud dabao result dekhne ke liye
# ============================================================
if demo_examples:
    st.subheader("⚡ Quick Demo — click to load real values into the form below")

    st.caption("🚨 Fraud Examples")
    fcols = st.columns(4)
    for i, ex in enumerate(demo_examples.get("fraud", [])):
        with fcols[i]:
            if st.button(f"Fraud #{i+1} ({ex['prob']*100:.0f}%)", key=f"fraud_demo_{i}"):
                load_preset(ex)
                st.rerun()

    st.caption("✅ Legitimate Examples")
    lcols = st.columns(4)
    for i, ex in enumerate(demo_examples.get("legit", [])):
        with lcols[i]:
            if st.button(f"Legit #{i+1} ({ex['prob']*100:.0f}%)", key=f"legit_demo_{i}"):
                load_preset(ex)
                st.rerun()

    st.divider()
else:
    st.info("💡 Tip: place `demo_examples.json` in this folder to enable one-click demo buttons.")

# ============================================================
# manual wala form - same fields hai, bas preset dabane pe pehle se fill ho jate hai
# ============================================================
st.subheader("🔍 Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Basic Details**")
    transaction_amt = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=32000.0, key="transaction_amt")
    card1 = st.number_input("Card1 - Card identifier code", min_value=0, max_value=20000, key="card1")
    card_type = st.selectbox("Card Network", ["visa", "mastercard", "discover", "american express"], key="card_type")
    card_category = st.selectbox("Card Category", ["debit", "credit", "debit or credit", "charge card"], key="card_category")
    email_domain = st.selectbox("Email Domain",
                                 ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
                                  "anonymous.com", "aol.com", "comcast.net", "other"],
                                 key="email_domain")
    addr1 = st.number_input("Billing Address Code", min_value=0.0, max_value=999.0, key="addr1")

with col2:
    st.markdown("**Transaction / Counting Features (C1-C14)**")
    c1 = st.number_input("C1 - Times card used", min_value=0, max_value=5000, key="c1")
    c2 = st.number_input("C2 - Unique addresses used", min_value=0, max_value=5000, key="c2")
    c4 = st.number_input("C4 - Suspicious tx count", min_value=0, max_value=2253, key="c4")
    c6 = st.number_input("C6 - Unique devices", min_value=0, max_value=2253, key="c6")
    c8 = st.number_input("C8 - Unique emails", min_value=0, max_value=3331, key="c8")
    c10 = st.number_input("C10 - Unique phones", min_value=0, max_value=3257, key="c10")
    c11 = st.number_input("C11 - Unique buyers", min_value=0, max_value=3188, key="c11")
    c13 = st.number_input("C13 - Total transactions", min_value=0, max_value=3000, key="c13")
    c14 = st.number_input("C14 - Unique banks used", min_value=0, max_value=1500, key="c14")

with col3:
    st.markdown("**Time, Device & Vesta (V) Signals**")
    hour = st.slider("Transaction Hour (0-23)", 0, 23, key="hour")
    product = st.selectbox("Product Category", ["W", "H", "C", "S", "R"], key="product")
    device_type = st.selectbox("Device Type", ["desktop", "mobile", "unknown"], key="device_type")

    st.caption("Vesta engineered risk signals")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        v29 = st.number_input("V29", min_value=0.0, max_value=20.0, key="v29")
        v51 = st.number_input("V51", min_value=0.0, max_value=20.0, key="v51")
        v70 = st.number_input("V70", min_value=0.0, max_value=20.0, key="v70")
        v91 = st.number_input("V91", min_value=0.0, max_value=20.0, key="v91")
        v279 = st.number_input("V279", min_value=0.0, max_value=20.0, key="v279")
    with vcol2:
        v280 = st.number_input("V280", min_value=0.0, max_value=20.0, key="v280")
        v302 = st.number_input("V302", min_value=0.0, max_value=20.0, key="v302")
        v303 = st.number_input("V303", min_value=0.0, max_value=20.0, key="v303")
        v304 = st.number_input("V304", min_value=0.0, max_value=20.0, key="v304")

    with st.expander("Advanced (Optional)"):
        d1 = st.number_input("D1 - Days since last tx", min_value=0, max_value=640, value=0)
        d4 = st.number_input("D4 - Account age (days)", min_value=0, max_value=640, value=0)
        d15 = st.number_input("D15 - Days since last login", min_value=0, max_value=879, value=0)

st.divider()

if st.button("🔍 Detect Fraud", width="stretch", type="primary"):

    # agar preset load kiya hai to real wali poori row (223 features) se start karo
    # (jo V-columns UI me dikhte nahi wo bhi sahi rahenge), fir jo form me abhi
    # dikh raha hai wo upar daal do - taki manual change bhi respect ho jaye
    if "preset_full" in st.session_state:
        base = pd.DataFrame([st.session_state["preset_full"]])
        features = base.reindex(columns=feature_names, fill_value=0)
    else:
        features = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    features['TransactionAmt'] = transaction_amt
    features['card1'] = card1
    features['ProductCD'] = product_map[product]
    features['card4'] = card4_map[card_type]
    features['card6'] = card6_map[card_category]
    features['addr1'] = addr1
    features['P_emaildomain'] = email_map[email_domain]
    features['DeviceType'] = device_map[device_type]
    features['C1'] = c1
    features['C2'] = c2
    features['C4'] = c4
    features['C6'] = c6
    features['C8'] = c8
    features['C10'] = c10
    features['C11'] = c11
    features['C13'] = c13
    features['C14'] = c14
    features['V29'] = v29
    features['V51'] = v51
    features['V70'] = v70
    features['V91'] = v91
    features['V279'] = v279
    features['V280'] = v280
    features['V302'] = v302
    features['V303'] = v303
    features['V304'] = v304
    features['D1'] = d1
    features['D4'] = d4
    features['D15'] = d15
    features['Transaction_Hour'] = hour
    features['Transaction_Day'] = pd.Timestamp.now().dayofweek
    features['High_Amount'] = 1 if transaction_amt > 500 else 0
    features['Night_Transaction'] = 1 if (hour >= 23 or hour <= 5) else 0
    features['Amt_Per_Card'] = transaction_amt / 101

    probability = model.predict_proba(features)[0][1]
    prediction = 1 if probability >= threshold else 0
    risk_score = probability * 100

    st.divider()
    rcol1, rcol2, rcol3 = st.columns(3)

    with rcol1:
        if prediction == 1:
            st.error(f"🚨 FRAUD DETECTED!\nFraud Probability: {probability*100:.1f}%")
        else:
            st.success(f"✅ LEGITIMATE Transaction!\nFraud Probability: {probability*100:.1f}%")

    with rcol2:
        st.info(f"💰 Amount: ${transaction_amt:.2f}\n\n🕐 Hour: {hour}:00\n\n📱 Device: {device_type}")

    with rcol3:
        if prediction == 1:
            st.error(f"⚠️ Risk Score: {risk_score:.1f}%\n🔴 HIGH RISK")
        else:
            if risk_score >= 50:
                st.warning(f"⚠️ Risk Score: {risk_score:.1f}%\n🟡 MEDIUM RISK")
            else:
                st.success(f"✅ Risk Score: {risk_score:.1f}%\n🟢 LOW RISK")

    # ============================================================
    # SHAP wala part - ye batayega model ne ye decision kyu liya
    # ============================================================
    st.divider()
    st.subheader("🔬 Why did the model decide this?")

    with st.spinner("Computing feature contributions..."):
        shap_values = explainer.shap_values(features)
        # binary classifier hai isliye shap_values kabhi list [class0, class1] hota hai
        # kabhi 3D array, dono case handle karna padega
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # contributions toward class 1 (fraud)
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

    contrib = pd.Series(sv, index=feature_names)
    top_contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#e05252" if v > 0 else "#2ecc71" for v in top_contrib.values[::-1]]
    ax.barh(top_contrib.index[::-1], top_contrib.values[::-1], color=colors)
    ax.set_xlabel("Contribution to Fraud Probability", fontsize=9)
    ax.set_title("Top 10 Features Driving This Prediction", fontsize=10)
    ax.axvline(0, color="white", linewidth=0.8)
    ax.tick_params(axis='both', labelsize=8, colors="white")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")
    plt.tight_layout()
    st.pyplot(fig, width="content")

    st.caption("🔴 Red bars push the prediction toward FRAUD · 🟢 Green bars push it toward LEGITIMATE")