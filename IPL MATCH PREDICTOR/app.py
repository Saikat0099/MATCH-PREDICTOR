import streamlit as st
import pickle
import pandas as pd

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="centered"
)

# ---------------------------
# Custom CSS (Premium UI)
# ---------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        font-size: 16px;
        color: gray;
        margin-bottom: 25px;
    }
    .card {
        padding: 18px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
        margin-bottom: 20px;
    }
    .vs-box {
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        padding: 10px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.06);
        margin-bottom: 20px;
    }
    .result-box-win {
        padding: 16px;
        border-radius: 16px;
        background: rgba(0, 255, 100, 0.10);
        border: 1px solid rgba(0, 255, 100, 0.25);
        font-size: 20px;
        font-weight: 700;
        text-align: center;
    }
    .result-box-loss {
        padding: 16px;
        border-radius: 16px;
        background: rgba(255, 0, 0, 0.10);
        border: 1px solid rgba(255, 0, 0, 0.25);
        font-size: 20px;
        font-weight: 700;
        text-align: center;
    }
    .summary-title {
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Data
# ---------------------------
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("pipe.pkl", "rb"))

pipe = load_model()

# ---------------------------
# Helper
# ---------------------------
def overs_to_balls(overs_float):
    overs_int = int(overs_float)
    balls_part = round((overs_float - overs_int) * 10)

    if balls_part > 5:
        return None

    return overs_int * 6 + balls_part

def safe_round(x):
    return int(round(x))

# ---------------------------
# Header UI
# ---------------------------
st.markdown('<div class="main-title">🏏 IPL Win Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Premium UI • Streamlit + Machine Learning</div>', unsafe_allow_html=True)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("⚙️ Match Inputs")

batting_team = st.sidebar.selectbox("Batting Team", sorted(teams))
bowling_team = st.sidebar.selectbox("Bowling Team", sorted(teams))
selected_city = st.sidebar.selectbox("Host City", sorted(cities))

st.sidebar.markdown("---")

target = st.sidebar.number_input("Target Score", min_value=1, step=1)
score = st.sidebar.number_input("Current Score", min_value=0, step=1)

overs = st.sidebar.number_input(
    "Overs Completed (Example: 14.3)",
    min_value=0.0,
    max_value=20.0,
    step=0.1
)

wickets_out = st.sidebar.number_input(
    "Wickets Out",
    min_value=0,
    max_value=10,
    step=1
)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔥 Predict Win Probability")

# ---------------------------
# Main UI
# ---------------------------
st.markdown(f'<div class="vs-box">{batting_team} 🆚 {bowling_team}</div>', unsafe_allow_html=True)

if predict_btn:

    if batting_team == bowling_team:
        st.error("❌ Batting and Bowling team cannot be same!")
        st.stop()

    balls_bowled = overs_to_balls(overs)

    if balls_bowled is None:
        st.error("❌ Invalid overs format. Example valid: 14.3, 5.2, 19.5")
        st.stop()

    if balls_bowled > 120:
        st.error("❌ Overs cannot be more than 20.")
        st.stop()

    runs_left = target - score
    balls_left = 120 - balls_bowled
    wickets_left = 10 - wickets_out

    if balls_left <= 0:
        st.error("❌ Match already finished.")
        st.stop()

    overs_completed = balls_bowled / 6
    crr = 0 if overs_completed == 0 else score / overs_completed
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "city": [selected_city],
        "runs_left": [runs_left],
        "balls_left": [balls_left],
        "wickets": [wickets_left],
        "total_runs_x": [target],
        "crr": [crr],
        "rrr": [rrr]
    })

    result = pipe.predict_proba(input_df)
    loss = float(result[0][0])
    win = float(result[0][1])

    win_pct = safe_round(win * 100)
    loss_pct = safe_round(loss * 100)

    # ---------------------------
    # Result Cards
    # ---------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏏 Batting Team", batting_team)
        st.metric("🟢 Win Chance", f"{win_pct}%")
        st.progress(win)

    with col2:
        st.metric("🎯 Bowling Team", bowling_team)
        st.metric("🔴 Win Chance", f"{loss_pct}%")
        st.progress(loss)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # Final Winner Box
    # ---------------------------
    if win > loss:
        st.markdown(f'<div class="result-box-win">🟢 {batting_team} is likely to WIN!</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box-loss">🔴 {bowling_team} is likely to WIN!</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------
    # Summary Box
    # ---------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="summary-title">🧾 Match Summary</div>', unsafe_allow_html=True)

    colA, colB, colC = st.columns(3)

    with colA:
        st.write(f"📍 **City:** {selected_city}")
        st.write(f"🎯 **Target:** {target}")
        st.write(f"🏏 **Score:** {score}/{wickets_out}")

    with colB:
        st.write(f"⏳ **Overs:** {overs}")
        st.write(f"🎾 **Balls Left:** {balls_left}")
        st.write(f"🧤 **Wickets Left:** {wickets_left}")

    with colC:
        st.write(f"⚡ **Runs Left:** {runs_left}")
        st.write(f"📈 **CRR:** {crr:.2f}")
        st.write(f"📉 **RRR:** {rrr:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Enter match details from sidebar and click **Predict Win Probability**.")