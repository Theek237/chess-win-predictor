import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Chess Win Predictor",
    page_icon="♟️",
    layout="centered"
)

ECO_FAMILIES = {
    'A': 'Flank Openings (A)',
    'B': 'Semi-Open Games (B)',
    'C': 'Open Games (C)',
    'D': 'Closed & Semi-Closed (D)',
    'E': 'Indian Defences (E)',
}

# Common popular time controls shown as friendly options
TIME_PRESETS = {
    "Bullet  1+0  (60s)":    "1+0",
    "Bullet  2+1  (120s)":   "2+1",
    "Blitz   3+0  (180s)":   "3+0",
    "Blitz   3+2  (260s)":   "3+2",
    "Blitz   5+0  (300s)":   "5+0",
    "Blitz   5+3  (420s)":   "5+3",
    "Rapid   10+0 (600s)":   "10+0",
    "Rapid   15+10(1000s)":  "15+10",
    "Classical 30+0(1800s)": "30+0",
    "Classical 60+0(3600s)": "60+0",
    "Custom...":              None,
}

def get_time_category_num(tc: str) -> int:
    """Return 0=Bullet, 1=Blitz, 2=Rapid, 3=Classical."""
    try:
        parts = tc.split('+')
        base = int(parts[0])
        inc  = int(parts[1]) if len(parts) > 1 else 0
        est  = base + inc * 40
        if est < 180:   return 0
        elif est < 600:  return 1
        elif est < 1800: return 2
        else:            return 3
    except Exception:
        return 1

def get_time_category_label(tc: str) -> str:
    return ["Bullet", "Blitz", "Rapid", "Classical"][get_time_category_num(tc)]

@st.cache_resource
def load_assets():
    model   = joblib.load('chess_rf_model.pkl')
    scaler  = joblib.load('scaler.pkl')
    le_eco  = joblib.load('le_eco.pkl')
    le_time = joblib.load('le_time.pkl')
    return model, scaler, le_eco, le_time

try:
    model, scaler, le_eco, le_time = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Detect model type and feature count for correct inference path
model_type   = type(model).__name__         # "XGBClassifier" or "RandomForestClassifier"
n_features   = model.n_features_in_ if hasattr(model, 'n_features_in_') else 5
model_label  = "XGBoost" if "XGB" in model_type else "Random Forest"

# --- Main UI ---
st.title("♟️ Chess Win Predictor")
st.markdown(
    f"Predict the outcome of a chess game using a **{model_label}** model "
    f"trained on **6M+ Lichess games**."
)
st.info("ℹ️ **Usage:** Enter both players' **Elo Ratings**, select an **ECO opening code**, and pick a **Time Control**.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚪ White Player")
    white_elo = st.number_input("White Elo Rating", min_value=500, max_value=3500, value=1500)
    eco_code  = st.selectbox(
        "Opening (ECO Code)",
        le_eco.classes_,
        index=list(le_eco.classes_).index('B00') if 'B00' in le_eco.classes_ else 0
    )

with col2:
    st.subheader("⚫ Black Player")
    black_elo    = st.number_input("Black Elo Rating", min_value=500, max_value=3500, value=1500)
    preset_label = st.selectbox("Time Control", list(TIME_PRESETS.keys()))
    preset_val   = TIME_PRESETS[preset_label]

    if preset_val is None:
        # Custom input
        time_control = st.text_input(
            "Enter time control (e.g. 300+5)",
            value="300+5",
            help="Format: seconds+increment, e.g. 300+5 means 5 minutes + 5 second increment"
        )
        # Validate format
        try:
            parts = time_control.split('+')
            int(parts[0])
            if len(parts) > 1: int(parts[1])
        except Exception:
            st.error("Invalid format. Use seconds+increment, e.g. 300+5")
            st.stop()
    else:
        time_control = preset_val

# Derived info display
elo_diff   = white_elo - black_elo
time_label = get_time_category_label(time_control)
eco_family = ECO_FAMILIES.get(eco_code[0].upper(), "Other") if eco_code else "Unknown"

col_a, col_b, col_c = st.columns(3)
col_a.metric("Elo Difference", f"{elo_diff:+d}", help="Positive = White advantage")
col_b.metric("Time Format",    time_label)
col_c.metric("Opening Family", eco_family)

st.divider()

if st.button("Predict Outcome", type="primary", use_container_width=True):
    st.write("---")

    # Safe encode — use a sensible fixed default instead of random first-char match
    def safe_encode(encoder, value, default):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        fallback = default if default in encoder.classes_ else encoder.classes_[0]
        st.warning(f"'{value}' not seen in training data — using default '{fallback}'.")
        return encoder.transform([fallback])[0]

    encoded_eco  = safe_encode(le_eco,  eco_code,     'B00')
    encoded_time = safe_encode(le_time, time_control, '300+0')

    elo_difference   = white_elo - black_elo
    abs_elo_diff     = abs(elo_difference)
    close_match      = 1 if abs_elo_diff < 100 else 0
    eco_family_num   = max(0, min(ord(eco_code[0].upper()) - ord('A'), 4))
    time_category_num = get_time_category_num(time_control)

    if n_features >= 9:
        input_data = pd.DataFrame({
            'WhiteElo':     [white_elo],
            'BlackElo':     [black_elo],
            'ECO':          [encoded_eco],
            'TimeControl':  [encoded_time],
            'EloDifference':[elo_difference],
            'AbsEloDiff':   [abs_elo_diff],
            'CloseMatch':   [close_match],
            'ECO_Family':   [eco_family_num],
            'TimeCategory': [time_category_num],
        })
    else:
        input_data = pd.DataFrame({
            'WhiteElo':     [white_elo],
            'BlackElo':     [black_elo],
            'ECO':          [encoded_eco],
            'TimeControl':  [encoded_time],
            'EloDifference':[elo_difference],
        })

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probs        = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

    st.subheader("📊 Match Prediction")

    if prediction == 1:
        st.success("🏆 **White is favored to Win!**")
    elif prediction == 0:
        st.error("🏆 **Black is favored to Win!**")
    else:
        st.warning("🤝 **The match is likely to end in a Draw.**")

    if probs is not None:
        max_prob = max(probs)
        if max_prob >= 0.55:
            confidence, conf_color = "High",   "🟢"
        elif max_prob >= 0.42:
            confidence, conf_color = "Medium",  "🟡"
        else:
            confidence, conf_color = "Low",    "🔴"

        st.markdown(f"**Confidence:** {conf_color} {confidence} ({max_prob*100:.1f}%)")

        if abs_elo_diff < 100:
            st.caption("Players are closely matched — draw is relatively more likely.")
        elif abs_elo_diff > 300:
            stronger = "White" if elo_difference > 0 else "Black"
            st.caption(f"Large Elo gap ({abs_elo_diff} pts) — {stronger} has a significant advantage.")

        st.markdown("**Probability Breakdown:**")
        for label, prob in [("⚫ Black Win", probs[0]), ("⚪ White Win", probs[1]), ("🤝 Draw", probs[2])]:
            st.markdown(f"{label}: **{prob*100:.1f}%**")
            st.progress(float(prob))
