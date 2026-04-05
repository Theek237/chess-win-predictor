import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Chess Win Predictor",
    page_icon="♟️",
    layout="centered"
)

# ECO opening family mapping (first letter of ECO code)
ECO_FAMILIES = {
    'A': 'Flank Openings (A)',
    'B': 'Semi-Open Games (B)',
    'C': 'Open Games (C)',
    'D': 'Closed & Semi-Closed (D)',
    'E': 'Indian Defences (E)',
}

def get_time_category(time_control: str) -> str:
    """Map raw time control string to Bullet/Blitz/Rapid/Classical."""
    try:
        base = int(time_control.split('+')[0])
        increment = int(time_control.split('+')[1]) if '+' in time_control else 0
        estimated = base + increment * 40  # rough estimate of total seconds
        if estimated < 180:
            return "Bullet"
        elif estimated < 600:
            return "Blitz"
        elif estimated < 1800:
            return "Rapid"
        else:
            return "Classical"
    except Exception:
        return "Unknown"

@st.cache_resource
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
st.markdown("Enter match details to predict the most likely outcome using a Random Forest model trained on **6M+ Lichess games**.")

st.info("ℹ️ **Usage:** Enter both players' **Elo Ratings** (500–3500), select an **ECO Code** (opening), and pick a **Time Control**.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚪ White Player")
    white_elo = st.number_input("White Elo Rating", min_value=500, max_value=3500, value=1500)
    eco_code = st.selectbox(
        "Opening (ECO Code)",
        le_eco.classes_,
        index=list(le_eco.classes_).index('B00') if 'B00' in le_eco.classes_ else 0
    )

with col2:
    st.subheader("⚫ Black Player")
    black_elo = st.number_input("Black Elo Rating", min_value=500, max_value=3500, value=1500)
    time_control = st.selectbox("Time Control Format", le_time.classes_)

# Show derived info to user
elo_diff = white_elo - black_elo
time_cat = get_time_category(str(time_control))
eco_family = ECO_FAMILIES.get(eco_code[0].upper(), "Other") if eco_code else "Unknown"

col_a, col_b, col_c = st.columns(3)
col_a.metric("Elo Difference", f"{elo_diff:+d}", help="Positive = White advantage")
col_b.metric("Time Format", time_cat)
col_c.metric("Opening Family", eco_family)

st.divider()

button_clicked = st.button("Predict Outcome", type="primary", use_container_width=True)

if button_clicked:
    st.write("---")

    # --- Safe encoding with fallback for unseen labels ---
    def safe_encode(encoder, value):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        # Fallback: find closest alphabetically
        classes = list(encoder.classes_)
        closest = min(classes, key=lambda x: abs(ord(x[0]) - ord(str(value)[0])))
        st.warning(f"'{value}' not seen during training — using closest match '{closest}'.")
        return encoder.transform([closest])[0]

    encoded_eco = safe_encode(le_eco, eco_code)
    encoded_time = safe_encode(le_time, time_control)

    elo_difference = white_elo - black_elo
    abs_elo_diff = abs(elo_difference)

    # Time category: 0=Bullet, 1=Blitz, 2=Rapid, 3=Classical
    time_cat_map = {"Bullet": 0, "Blitz": 1, "Rapid": 2, "Classical": 3, "Unknown": 1}
    time_category_num = time_cat_map.get(get_time_category(str(time_control)), 1)

    # ECO family: A=0, B=1, C=2, D=3, E=4
    eco_family_num = ord(eco_code[0].upper()) - ord('A') if eco_code else 1
    eco_family_num = max(0, min(eco_family_num, 4))  # clamp to 0-4

    close_match = 1 if abs_elo_diff < 100 else 0

    # Detect whether the loaded model expects 5 or 9 features
    n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 5

    if n_features >= 9:
        input_data = pd.DataFrame({
            'WhiteElo': [white_elo],
            'BlackElo': [black_elo],
            'ECO': [encoded_eco],
            'TimeControl': [encoded_time],
            'EloDifference': [elo_difference],
            'AbsEloDiff': [abs_elo_diff],
            'CloseMatch': [close_match],
            'ECO_Family': [eco_family_num],
            'TimeCategory': [time_category_num],
        })
    else:
        input_data = pd.DataFrame({
            'WhiteElo': [white_elo],
            'BlackElo': [black_elo],
            'ECO': [encoded_eco],
            'TimeControl': [encoded_time],
            'EloDifference': [elo_difference],
        })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

    # --- Outcome display ---
    st.subheader("📊 Match Prediction")

    outcome_labels = {0: "Black", 1: "White", 2: "Draw"}
    outcome_label = outcome_labels.get(prediction, "Unknown")

    if prediction == 1:
        st.success("🏆 **White is favored to Win!**")
    elif prediction == 0:
        st.error("🏆 **Black is favored to Win!**")
    else:
        st.warning("🤝 **The match is likely to end in a Draw.**")

    # --- Confidence level ---
    if probs is not None:
        max_prob = max(probs)
        if max_prob >= 0.55:
            confidence = "High"
            conf_color = "🟢"
        elif max_prob >= 0.42:
            confidence = "Medium"
            conf_color = "🟡"
        else:
            confidence = "Low"
            conf_color = "🔴"

        st.markdown(f"**Confidence:** {conf_color} {confidence} ({max_prob*100:.1f}%)")

        # Context hint
        if abs_elo_diff < 100:
            st.caption("Players are closely matched — draw is relatively more likely.")
        elif abs_elo_diff > 300:
            stronger = "White" if elo_difference > 0 else "Black"
            st.caption(f"Large Elo gap ({abs_elo_diff} pts) — {stronger} has a significant advantage.")

        # Probability breakdown
        st.markdown("**Probability Breakdown:**")
        prob_data = pd.DataFrame({
            "Outcome": ["⚫ Black Win", "⚪ White Win", "🤝 Draw"],
            "Probability": [f"{probs[0]*100:.1f}%", f"{probs[1]*100:.1f}%", f"{probs[2]*100:.1f}%"],
            "Bar": [probs[0], probs[1], probs[2]]
        })

        for _, row in prob_data.iterrows():
            st.markdown(f"{row['Outcome']}: **{row['Probability']}**")
            st.progress(float(row['Bar']))
