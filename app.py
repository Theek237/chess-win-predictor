import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model and preprocessors
@st.cache_resource # Caches the models so they aren't reloaded on every interaction
def load_assets():
    model = joblib.load('chess_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_eco = joblib.load('le_eco.pkl')
    le_time = joblib.load('le_time.pkl')
    return model, scaler, le_eco, le_time

model, scaler, le_eco, le_time = load_assets()

# 2. Build the UI
st.title("♟️ Chess Win Predictor")
st.markdown("Predict the outcome of a chess match based on player ratings and game settings.")

col1, col2 = st.columns(2)

with col1:
    white_elo = st.number_input("White Player Elo", min_value=500, max_value=3500, value=1500)
    # Use the original classes from the encoder for the dropdown
    eco_code = st.selectbox("Opening ECO Code (e.g., C20, B00)", le_eco.classes_)

with col2:
    black_elo = st.number_input("Black Player Elo", min_value=500, max_value=3500, value=1500)
    time_control = st.selectbox("Time Control", le_time.classes_)

# 3. Handle Prediction
if st.button("Predict Outcome"):
    # Calculate derived feature
    elo_difference = white_elo - black_elo
    
    # Encode categorical inputs
    encoded_eco = le_eco.transform([eco_code])[0]
    encoded_time = le_time.transform([time_control])[0]
    
    # Assemble into a DataFrame matching the training data structure
    # Note: Ensure this matches the exact column order of `X` in your notebook
    input_data = pd.DataFrame({
        'WhiteElo': [white_elo],
        'BlackElo': [black_elo],
        'EloDifference': [elo_difference],
        'ECO': [encoded_eco],
        'TimeControl': [encoded_time]
    })
    
    # Scale the features
    input_scaled = scaler.transform(input_data)
    
    # Predict!
    prediction = model.predict(input_scaled)[0]
    
    # Display Results
    st.divider()
    if prediction == 1:
        st.success("🏆 Prediction: **White Wins!**")
    elif prediction == 0:
        st.error("🏆 Prediction: **Black Wins!**")
    else:
        st.info("🤝 Prediction: **Draw**")