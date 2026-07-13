import streamlit as st
import numpy as np
import joblib

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_housing_model.pkl")

model = load_model()

# --------------------------------------------------
# CUSTOM CSS (BIG & BOLD SIDEBAR)
# --------------------------------------------------
st.markdown("""
<style>

/* Sidebar width */
section[data-testid="stSidebar"] {
    width: 380px !important;
}

/* Sidebar background */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Sidebar title */
.sidebar-title {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 10px;
    color: #ffffff;
}

/* Sidebar subtitle */
.sidebar-sub {
    font-size: 16px;
    color: #d1d1d1;
    margin-bottom: 20px;
}

/* Input labels */
label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: white !important;
}

/* Main title */
.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #1f4aff;
}

/* Card styling */
.card {
    padding: 25px;
    border-radius: 18px;
    background-color: #f9f9f9;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="main-title">🏠 AI Housing Price Predictor</div>', unsafe_allow_html=True)
st.markdown("### Smart house price estimation using Machine Learning")
st.markdown("---")

# --------------------------------------------------
# SIDEBAR (BIG & BOLD)
# --------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">📊 Property Inputs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Enter details to predict house price</div>', unsafe_allow_html=True)

    avg_income = st.number_input(
        "Average Area Income ($)",
        10000.0, 200000.0, 50000.0, step=1000.0
    )

    house_age = st.slider(
        "Average House Age (Years)",
        1.0, 20.0, 5.0
    )

    rooms = st.slider(
        "Number of Rooms",
        1.0, 10.0, 5.0
    )

    bedrooms = st.slider(
        "Number of Bedrooms",
        1.0, 6.0, 3.0
    )

    population = st.number_input(
        "Area Population",
        1000.0, 100000.0, 30000.0, step=500.0
    )

# --------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🏘️ Property Overview")
    st.write(f"**💰 Income Level:** ${avg_income:,.0f}")
    st.write(f"**🏠 House Age:** {house_age} years")
    st.write(f"**🛏️ Rooms / Bedrooms:** {rooms} / {bedrooms}")
    st.write(f"**👥 Population:** {population:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Prediction Result")

    if st.button("🔮 Predict House Price"):
        input_data = np.array([[avg_income, house_age, rooms, bedrooms, population]])
        prediction = model.predict(input_data)[0]

        st.metric("Estimated Price", f"${prediction:,.2f}")

        if prediction < 500000:
            st.info("💡 Affordable Housing")
        elif prediction < 1000000:
            st.success("🏡 Mid-Range Housing")
        else:
            st.warning("🏰 Luxury Housing")

        st.progress(85)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
---
### 🚀 About This Project
AI-powered house price prediction system built using Machine Learning & Streamlit.

👨‍💻 Developed by **Ankit Kumar**
""")
