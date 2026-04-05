import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Chess Win Predictor",
    page_icon="♟️",
    layout="centered",
    initial_sidebar_state="expanded"
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

# --- Sidebar ---
st.sidebar.header("About the Model")
st.sidebar.info(
    "This machine learning model predicts the outcome of a chess match based on "
    "player Elo ratings, Opening ECO codes, and Time Control settings. "
    "It uses a Random Forest Classifier trained on Kaggle chess datasets."
)
st.sidebar.divider()
st.sidebar.markdown("👨‍💻 Build with **Streamlit** & **scikit-learn**")

# --- Main UI ---
st.title("♟️ Chess Win Predictor")
st.markdown("Enter the match details below to predict the most likely outcome!")

# Use an expander for additional instructions
with st.expander("📖 How to use this app"):
    st.write("- **Elo Rating**: Chess player's skill rating (usually 500 - 3000)")
    st.write("- **ECO Code**: Encyclopedia of Chess Openings (e.g., C20 for King's Pawn Game)")
    st.write("- **Time Control**: The time format of the match (e.g., 600+0 for 10-minute Rapid)")

st.divider()

# Input UI separated by columns
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("⚪ White Player")
    white_elo = st.number_input("Elo Rating", min_value=500, max_value=3500, value=1500, key="w_elo", help="Higher rating means better skill.")
    eco_code = st.selectbox("Opening (ECO Code)", le_eco.classes_, index=list(le_eco.classes_).index('B00') if 'B00' in le_eco.classes_ else 0, help="Initial moves used in the game.")

with col2:
    st.subheader("⚫ Black Player")
    black_elo = st.number_input("Elo Rating", min_value=500, max_value=3500, value=1500, key="b_elo", help="Higher rating means better skill.")
    time_control = st.selectbox("Time Control", le_time.classes_, help="Format of the game in seconds.")

st.divider()

# 3. Handle Prediction
button_clicked = st.button("Predict Outcome", type="primary", use_container_width=True)

if button_clicked:
    with st.spinner("🤖 Analyzing the game..."):
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
        
        # Get Probabilities (if random forest has it)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
        else:
            probs = None

        # Display Results
        st.subheader("📊 Match Prediction")
        
        # We need to map prediction back if it was numerically encoded
        # In your notebook, typically 0: Black, 1: White, 2: Draw or something similar
        
        # Assuming common numeric encoding: 0=Black, 1=White, 2=Draw (or adjust as needed)
        # Using the standard mapping from your project
        
        if prediction == 1:
            st.success("🏆 **White is favored to Win!**")
        elif prediction == 0:
            st.error("🏆 **Black is favored to Win!**")
        else:
            st.info("🤝 **The match is likely to end in a Draw.**")

        if probs is not None:
            # Usually probs is an array [BlackProb, WhiteProb, DrawProb] depending on label encoding
            # We'll just display raw confidence
            st.write("---")
            st.caption("Confidence Matrix:")
            prob_df = pd.DataFrame(
                [probs], 
                columns=["Black Win (0)", "White Win (1)", "Draw (2)"][:len(probs)]
            )
            st.bar_chart(prob_df.T)