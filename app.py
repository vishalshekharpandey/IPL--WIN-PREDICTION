
import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set page layout
st.set_page_config(page_title="IPL Win Predictor", layout="centered")
st.title("🏏 IPL Win Predictor")

# Teams
teams = sorted([
    'Chennai Super Kings',
    'Delhi Capitals',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
])

# Cities
cities = sorted([
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
])

# UI Layout
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select the batting team", teams)

with col2:
    bowling_team = st.selectbox("Select the bowling team", [team for team in teams if team != batting_team])

selected_city = st.selectbox("Select Match City", cities)

target = st.number_input("Target", min_value=0)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input("Score", min_value=0)

with col4:
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=9)

with col5:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)

# Prediction
if st.button("Predict Probability"):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    loss_prob = round(result[0][0] * 100)
    win_prob = round(result[0][1] * 100)

    st.subheader("📊 Winning Probability")
    st.success(f"✅ {batting_team}: {win_prob}%")
    st.error(f"❌ {bowling_team}: {loss_prob}%")
