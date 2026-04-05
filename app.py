import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Chess Win Predictor",
    page_icon="♟️",
    layout="centered"
)

# 1. Load the saved model and preprocessors
@st.cache_resource # Caches the models so they aren't reloaded on every interaction
def load_assets():
    model = joblib.load('chess_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_eco = joblib.load('le_eco.pkl')
    le_time = joblib.load('le_time.pkl')
    return model, scaler, le_eco, le_time

try:
    model, scaler, le_eco, le_time = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- Main UI ---
st.title("♟️ Chess Win Predictor")
st.markdown("Enter the match details below to predict the most likely outcome based on our Random Forest Machine Learning model!")

# Informational box instead of sidebar or expanders
st.info("ℹ️ **Usage:** Enter both players' **Elo Ratings** (500–3500), select an **ECO Code** (e.g., C20 for King's Pawn Game), and pick a **Time Control**.")

st.divider()

# Input UI separated by columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚪ White Player")
    white_elo = st.number_input("White Elo Rating", min_value=500, max_value=3500, value=1500)
    eco_code = st.selectbox("Opening (ECO Code)", le_eco.classes_, index=list(le_eco.classes_).index('B00') if 'B00' in le_eco.classes_ else 0)

with col2:
    st.subheader("⚫ Black Player")
    black_elo = st.number_input("Black Elo Rating", min_value=500, max_value=3500, value=1500)
    time_control = st.selectbox("Time Control Format", le_time.classes_)

st.divider()

# 3. Handle Prediction
button_clicked = st.button("Predict Outcome", type="primary", use_container_width=True)

if button_clicked:
    st.write("---")
    # Calculate derived feature
    elo_difference = white_elo - black_elo
    
    # Encode categorical inputs
    encoded_eco = le_eco.transform([eco_code])[0]
    encoded_time = le_time.transform([time_control])[0]
    
    # Assemble into a DataFrame matching the EXACT training data structure
    input_data = pd.DataFrame({
        'WhiteElo': [white_elo],
        'BlackElo': [black_elo],
        'ECO': [encoded_eco],
        'TimeControl': [encoded_time],
        'EloDifference': [elo_difference]
    })
    
    # Scale the features
    input_scaled = scaler.transform(input_data)
    
    # Predict class
    prediction = model.predict(input_scaled)[0]
    
    # Display Results
    st.subheader("📊 Match Prediction")
    
    if prediction == 1:
        st.success("🏆 **White is favored to Win!**")
    elif prediction == 0:
        st.error("🏆 **Black is favored to Win!**")
    else:
        st.warning("🤝 **The match is likely to end in a Draw.**")

    # Get Probabilities (if random forest has it)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_scaled)[0]
        
        # Determine number of classes to show safely
        labels = ["Black Win (0)", "White Win (1)", "Draw (2)"][:len(probs)]
        
        # Display as a table to keep it very simple and stable
        prob_data = {
            "Outcome": labels,
            "Probability": [f"{p*100:.1f}%" for p in probs]
        }
        st.table(pd.DataFrame(prob_data))