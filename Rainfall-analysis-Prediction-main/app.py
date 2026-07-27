import streamlit as st
import pickle
import numpy as np
import pandas as pd
from annual_llm import get_annual_ai_suggestion
from llm import get_ai_suggestion

# ================================
# Page Config
# ================================
st.set_page_config(
    page_title="Rainfall Predictor India",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded")

# ================================
# Custom CSS
# ================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 40%, #0a2b45 100%);
    font-family: 'DM Sans', sans-serif;
}

h1 {
    font-family: 'Playfair Display', serif !important;
    color: #e8f4fd !important;
    font-size: 2.5rem !important;
    text-shadow: 0 0 30px rgba(100,180,255,0.4);
}

h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #b8d9f0 !important;
}

/* Selectbox */
.stSelectbox [data-baseweb="select"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(100,180,255,0.3) !important;
    border-radius: 10px !important;
}

.stSelectbox [data-baseweb="select"] * {
    color: white !important;
    -webkit-text-fill-color: white !important;
}

/* Number Input */
.stNumberInput input {
    background-color: #1a2238 !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
    caret-color: white !important;
    border: 1px solid #4da6ff !important;
} 

.stButton > button {
    background: linear-gradient(135deg, #1a6bb5, #0d9488) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(26,107,181,0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(26,107,181,0.5) !important;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(100,180,255,0.2) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

[data-testid="metric-container"] label {
    color: #7db8d9 !important;
    font-size: 0.85rem !important;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e8f4fd !important;
    font-family: 'Playfair Display', serif !important;
}

[data-testid="stSidebar"] {
    background: rgba(10,22,40,0.95) !important;
    border-right: 1px solid rgba(100,180,255,0.15) !important;
}

.stSuccess {
    background: rgba(16,185,129,0.15) !important;
    border: 1px solid rgba(16,185,129,0.4) !important;
    border-radius: 12px !important;
}

.stWarning {
    background: rgba(245,158,11,0.15) !important;
    border: 1px solid rgba(245,158,11,0.4) !important;
    border-radius: 12px !important;
}

.stError {
    background: rgba(239,68,68,0.15) !important;
    border: 1px solid rgba(239,68,68,0.4) !important;
    border-radius: 12px !important;
}

.stInfo {
    background: rgba(59,130,246,0.1) !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    border-radius: 12px !important;
    color: #b8d9f0 !important;
}

label { color: #7db8d9 !important; font-weight: 500 !important; }
p { color: #b8d9f0 !important; }
hr { border-color: rgba(100,180,255,0.15) !important; }

.rain-drops {
    font-size: 1.5rem;
    animation: fall 2s infinite;
    display: inline-block;
    margin: 0 4px;
} 

@keyframes fall {
    0% { transform: translateY(-5px); opacity: 0.5; }
    50% { transform: translateY(5px); opacity: 1; }
    100% { transform: translateY(-5px); opacity: 0.5; }
}
/* Fix Selectbox Text */
div[data-baseweb="select"] span {
    color: white !important;
}

div[role="listbox"] {
    background-color: #1b263b !important;
}

div[role="option"] {
    color: white !important;
    background-color: #1b263b !important;
}

div[role="option"]:hover {
    background-color: #2c5282 !important;
} 
</style>
""", unsafe_allow_html=True)

# ================================
# Load Models
# ================================
import os
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
@st.cache_resource
def load_models():
    lr = pickle.load(open( os.path.join(BASE_DIR, 'lr_model.pkl'),      'rb'))
    l1 = pickle.load(open( os.path.join(BASE_DIR, 'lr_model1.pkl'),     'rb'))
    sc = pickle.load(open( os.path.join(BASE_DIR, 'sc_scaler.pkl'),     'rb'))
    ss = pickle.load(open( os.path.join(BASE_DIR, 'ss_scaler.pkl'),     'rb'))
    le = pickle.load(open( os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb'))

    medians_df = pd.read_csv('region_medians.csv')

    if 'SUBDIVISION' in medians_df.columns:
        medians_df = medians_df.set_index('SUBDIVISION') 

    medians_df.columns = medians_df.columns.str.strip()
    medians_df.index   = medians_df.index.str.strip()

    return lr, l1, sc, ss, le, medians_df

lr, l1, sc, ss, le, medians_df = load_models()

# ================================
# Constants
# ================================
regions = [
    'ANDAMAN & NICOBAR ISLANDS',
    'ARUNACHAL PRADESH',
    'ASSAM & MEGHALAYA',
    'NAGA MANI MIZO TRIPURA',
    'SUB HIMALAYAN WEST BENGAL & SIKKIM',
    'GANGETIC WEST BENGAL',
    'ORISSA', 'JHARKHAND', 'BIHAR',
    'EAST UTTAR PRADESH',
    'WEST UTTAR PRADESH',
    'UTTARAKHAND',
    'HARYANA DELHI & CHANDIGARH',
    'PUNJAB', 'HIMACHAL PRADESH',
    'JAMMU & KASHMIR',
    'WEST RAJASTHAN', 'EAST RAJASTHAN',
    'WEST MADHYA PRADESH', 'EAST MADHYA PRADESH',
    'GUJARAT REGION', 'SAURASHTRA & KUTCH',
    'KONKAN & GOA', 'MADHYA MAHARASHTRA',
    'MATATHWADA', 'VIDARBHA', 'CHHATTISGARH',
    'COASTAL ANDHRA PRADESH', 'TELANGANA',
    'RAYALSEEMA', 'TAMIL NADU',
    'COASTAL KARNATAKA', 'NORTH INTERIOR KARNATAKA',
    'SOUTH INTERIOR KARNATAKA', 'KERALA',
    'LAKSHADWEEP']

months = ['JAN', 'FEB', 'MAR', 'APR',
          'MAY', 'JUN', 'JUL', 'AUG',
          'SEP', 'OCT', 'NOV', 'DEC']

# ================================
# Helper Functions
# ================================
def get_flood_risk(annual):
    if annual > 1500:
        return "🔴 HIGH FLOOD RISK!", "error"
    elif annual > 1000:
        return "🟡 MEDIUM FLOOD RISK", "warning"
    else:
        return "🟢 LOW FLOOD RISK", "success"

def get_medians(region):
    try:
        mv = medians_df.loc[
            region.strip(), months].values.tolist()
        mv = [float(v) if (0 <= float(v) <= 3000)
              else 50.0 for v in mv]
        return mv
    except:
        return [50.0] * 12

def predict_monthly(region, mv):
    """
    X columns from Colab (EXACT ORDER) →
    JAN, FEB, MAR, APR, MAY, JUN,
    JUL, AUG, SEP, OCT, NOV, DEC,
    Jan-Feb, Mar-May, Jun-Sep, Oct-Dec,
    SUBDIVISION_encoded ← at END!
    """
    region_encoded = le.transform([region])[0]

    # Calculate quarterly from monthly values
    jan_feb = mv[0] + mv[1]
    mar_may = mv[2] + mv[3] + mv[4]
    jun_sep = mv[5] + mv[6] + mv[7] + mv[8]
    oct_dec = mv[9] + mv[10] + mv[11]

    # Build input with exact column order!
    input_df = pd.DataFrame(
        [mv + [jan_feb, mar_may,
               jun_sep, oct_dec,
               region_encoded]],
        columns=[
            'JAN', 'FEB', 'MAR', 'APR',
            'MAY', 'JUN', 'JUL', 'AUG',
            'SEP', 'OCT', 'NOV', 'DEC',
            'Jan-Feb', 'Mar-May',
            'Jun-Sep', 'Oct-Dec',
            'SUBDIVISION_encoded'])  # END!

    input_scaled = sc.transform(input_df)
    pred = lr.predict(input_scaled)[0]
    pred = max(200.0, min(pred, 5000.0))
    return pred, jan_feb, mar_may, jun_sep, oct_dec

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0'>
        <div style='font-size:3rem'>🌧️</div>
        <div style='font-family:Playfair Display,serif;
                    color:#e8f4fd; font-size:1.2rem;
                    margin-top:8px; font-weight:bold'>
            Rainfall Predictor
        </div>
        <div style='color:#7db8d9; font-size:0.8rem;
                    margin-top:4px'>
            India 
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "📍 Navigate",
        ["🌦️ Monthly Prediction",
         "📅 Next Year Prediction"])

    st.markdown("---")

    st.markdown("""
    <div style='color:#7db8d9; font-size:0.82rem;
                padding:10px; line-height:2'>
        <b style='color:#b8d9f0'>📊 Model Info</b><br>
        Algorithm : Linear Regression<br>
        Data      : 114 Years IMD<br>
        R² Score  : 0.85<br>
        Regions   : 36 Subdivisions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='color:#7db8d9; font-size:0.78rem;
                text-align:center; line-height:2'>
        ⚠️ <b style='color:#b8d9f0'>Flood Thresholds</b><br>
        🔴 HIGH &gt; 1500mm<br>
        🟡 MEDIUM 1000–1500mm<br>
        🟢 LOW &lt; 1000mm
    </div>
    """, unsafe_allow_html=True)

# ================================
# PAGE 1 - Monthly Prediction
# ================================
if page == "🌦️ Monthly Prediction":

    st.markdown("""
    <div style='text-align:center; padding:10px 0'> 
        <span class='rain-drops'>💧</span>
        <span class='rain-drops'
              style='animation-delay:0.3s'>💧</span>
        <span class='rain-drops'
              style='animation-delay:0.6s'>💧</span>
    </div>
    """, unsafe_allow_html=True)

    st.title("🌧️ Monthly Rainfall Predictor")
    st.markdown(
        "<p>Select your region and month, "
        "enter rainfall amount to predict "
        "annual rainfall & flood risk</p>",
        unsafe_allow_html=True)
    st.markdown("---")

    # Row 1 - Region, Month, Rainfall
    col1, col2, col3 = st.columns(3)

    with col1:
        region = st.selectbox(
            "📍 Select Region",
            regions,
            index=regions.index('KERALA'))

    with col2:
        month = st.selectbox(
            "📅 Select Month",
            months)

    with col3:
        rainfall = st.number_input(
            f"🌧️ {month} Rainfall (mm)",
            min_value=0.0,
            max_value=3000.0, 
            value=450.0,
            step=0.1)

    st.markdown("---")

    if st.button("🔍 Predict Annual Rainfall",
                 use_container_width=True):

        # Get region medians as base
        mv = get_medians(region)

        # Override selected month only
        mv[months.index(month)] = float(rainfall)

        # Predict
        annual_pred, jan_feb, mar_may, \
            jun_sep, oct_dec = \
            predict_monthly(region, mv)

        # Results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        col4, col5 = st.columns(2)
        with col4:
            st.metric(
                "🌧️ Predicted Annual Rainfall",
                f"{annual_pred:.1f} mm")
            st.progress(
                min(annual_pred / 4000, 1.0))

        with col5:
            risk, risk_type = \
                get_flood_risk(annual_pred)
            if risk_type == "error":
                st.error(f"### {risk}")
            elif risk_type == "warning":
                st.warning(f"### {risk}")
            else:
                st.success(f"### {risk}")

        # Quarterly breakdown
        st.markdown("---")
        st.markdown("### 📈 Quarterly Breakdown")

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("❄️ Jan-Feb",
                      f"{jan_feb:.1f}mm")
        with q2:
            st.metric("🌸 Mar-May",
                      f"{mar_may:.1f}mm")
        with q3:
            st.metric("⛈️ Jun-Sep",
                      f"{jun_sep:.1f}mm")
        with q4:
            st.metric("🍂 Oct-Dec",
                      f"{oct_dec:.1f}mm")

        # Summary
        st.markdown("---") 
        st.info(f"""
        📋 *Prediction Summary*
        - *Region* : {region}
        - *Month*  : {month} = {rainfall:.1f}mm
        - *Annual* : {annual_pred:.1f}mm
        - *Risk*   : {risk}
        """) 
 # ================================
# AI Agricultural Advisor
# ================================

        st.markdown("---") 
        st.subheader("🤖 AI Agricultural Advisor")


        with st.spinner("Generating AI recommendations..."):

            ai_response = get_ai_suggestion(
        region=region,
        month=month,
        annual_rainfall=annual_pred,
        monthly_rainfall=rainfall
        )

        st.success("✅ AI Recommendation Generated")

        st.markdown(ai_response)        

        

# ================================
# PAGE 2 - Next Year Prediction
# ================================
else:

    st.markdown("""
    <div style='text-align:center; padding:10px 0'>
        <span class='rain-drops'>📅</span>
    </div>
    """, unsafe_allow_html=True)

    st.title("📅 Next Year Rainfall Predictor")
    st.markdown(
        "<p>Enter this year's annual rainfall "
        "to predict next year's rainfall "
        "& flood risk</p>",
        unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        region2 = st.selectbox(
            "📍 Select Region",
            regions,
            index=regions.index('KERALA'))
    with col2:
        current_rainfall = st.number_input(
            "🌧️ This Year's Annual Rainfall (mm)",
            min_value=0.0,
            max_value=6000.0,
            value=3000.0,
            step=0.1)

    st.markdown("---")

    if st.button("🔍 Predict Next Year Rainfall",
                 use_container_width=True):

        region_encoded2 = le.transform(
            [region2])[0]

        input_df2 = pd.DataFrame(
            [[region_encoded2,
              float(current_rainfall)]],
            columns=[
                'SUBDIVISION_encoded',
                'ANNUAL'])

        input_scaled2 = ss.transform(input_df2)
        next_year     = l1.predict(input_scaled2)[0]
        next_year     = max(200.0,
                            min(next_year, 5000.0))
        change        = next_year - current_rainfall

        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("📅 This Year",
                      f"{current_rainfall:.1f}mm")
        with col4:
            st.metric("🌧️ Next Year Predicted",
                      f"{next_year:.1f}mm",
                      delta=f"{change:+.1f}mm")
        with col5:
            risk2, risk_type2 = \
                get_flood_risk(next_year)
            if risk_type2 == "error":
                st.error(f"### {risk2}")
            elif risk_type2 == "warning":
                st.warning(f"### {risk2}")
            else:
                st.success(f"### {risk2}")

        st.markdown("---")
        col6, col7 = st.columns(2)
        with col6:
            st.markdown("*📊 This Year*")
            st.progress(
                min(current_rainfall / 4000,
                    1.0))
        with col7:
            st.markdown(
                "*📊 Next Year (Predicted)*")
            st.progress(
                min(next_year / 4000, 1.0))

        st.markdown("---")
        trend = "📈 Increasing" \
            if change > 0 else "📉 Decreasing"
        st.info(f"""
        📋 *Prediction Summary*
        - *Region*     : {region2}
        - *This Year*  : {current_rainfall:.1f}mm
        - *Next Year*  : {next_year:.1f}mm
        - *Change*     : {change:+.1f}mm ({trend})
        - *Flood Risk* : {risk2}
        """)
        # ================================
        # AI Agricultural Advisor (Annual)
        # ================================
        st.markdown("---")
        st.subheader("🤖 AI Agricultural Advisor")

        with st.spinner("Generating AI recommendations..."):
            ai_response = get_annual_ai_suggestion(
                region=region2,
                current_rainfall=float(current_rainfall),
                predicted_annual_rainfall=float(next_year)
            )

        st.success("✅ AI Recommendation Generated")
        st.markdown(ai_response)