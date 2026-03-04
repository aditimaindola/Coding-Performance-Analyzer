import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import requests
from streamlit_lottie import st_lottie
from datetime import datetime, timedelta

from build_dataset import fetch_data, build_dataframe, compute_topic_metrics

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Coding Performance Analyzer", layout="wide")

# ---------------- THEME TOGGLE ----------------
theme = st.sidebar.toggle("🌙 Dark Mode", value=True)

background = "#0e1117" if theme else "#ffffff"
text_color = "white" if theme else "black"

st.markdown(f"""
<style>
html, body, [class*="css"] {{
    font-family: "Times New Roman", Times, serif;
}}

body {{
    background-color: {background};
    color: {text_color};
}}

.main-title {{
    font-size: 42px;
    font-weight: 700;
}}

.stMetric {{
    text-align: center;
}}

.stButton>button {{
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    font-size: 16px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- LOTTIE ----------------
def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie = load_lottie("https://assets2.lottiefiles.com/packages/lf20_tno6cg2w.json")

# ---------------- HEADER ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="main-title">🚀 Coding Performance Analyzer</div>', unsafe_allow_html=True)
    st.write("AI-powered analytics for your competitive programming journey.")

with col2:
    if lottie:
        st_lottie(lottie, height=150)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model/performance_model.pkl")

# ---------------- INPUT ----------------
username = st.text_input("Enter Codeforces Username")

if st.button("Analyze Performance"):

    with st.spinner("Analyzing..."):

        # ---------------- FETCH USER INFO ----------------
        info_url = f"https://codeforces.com/api/user.info?handles={username}"
        info_response = requests.get(info_url).json()

        if info_response["status"] != "OK":
            st.error("Invalid username!")
            st.stop()

        user_data = info_response["result"][0]
        rating = user_data.get("rating", "Unrated")
        max_rating = user_data.get("maxRating", "N/A")
        rank = user_data.get("rank", "N/A")
        avg_accuracy = 0

        # ---------------- FETCH SUBMISSIONS ----------------
        submissions = fetch_data(username)
        df = build_dataframe(submissions)
        topic_stats = compute_topic_metrics(df)

        if topic_stats.empty:
            st.error("No data found!")
            st.stop()

        # ---------------- ML PREDICTIONS ----------------
        X = topic_stats[["accuracy", "avg_difficulty", "total_attempts"]]
        topic_stats["prediction"] = model.predict(X)

        weak_topics = topic_stats[topic_stats["prediction"] == 0]
        strong_topics = topic_stats[topic_stats["prediction"] == 1]
        avg_accuracy = round(topic_stats["accuracy"].mean(), 2)

        # =====================================================
        # ================= PROFILE SECTION ===================
        # =====================================================
        st.markdown("## 👤 Profile Overview")

        colA, colB = st.columns([1, 2])

        with colA:
            st.metric("Current Rating", rating)
            st.metric("Max Rating", max_rating)

        with colB:
            st.markdown(f"""
            **Rank:** {rank.title()}  
            **Average Accuracy:** {avg_accuracy} %
            """)

        # =====================================================
        # ================= RATING GROWTH =====================
        # =====================================================
        st.markdown("## 📈 Rating Growth")

        rating_url = f"https://codeforces.com/api/user.rating?handle={username}"
        rating_response = requests.get(rating_url).json()

        if rating_response["status"] == "OK" and rating_response["result"]:
            rating_df = pd.DataFrame(rating_response["result"])
            rating_df["date"] = pd.to_datetime(
                rating_df["ratingUpdateTimeSeconds"], unit="s"
            )

            fig_rating = px.line(
                rating_df,
                x="date",
                y="newRating",
                title="Rating Progress Over Time",
            )
            fig_rating.update_traces(line=dict(width=3))
            st.plotly_chart(fig_rating, use_container_width=True)
        else:
            st.info("No rating history found.")

        # =====================================================
        # ================= LAST 30 DAYS ACTIVITY =============
        # =====================================================
        st.markdown("## 🔥 Last 30 Days Activity")

        solved_submissions = [
            sub for sub in submissions if sub.get("verdict") == "OK"
        ]

        if solved_submissions:

            dates = [
                datetime.fromtimestamp(sub["creationTimeSeconds"]).date()
                for sub in solved_submissions
            ]

            streak_df = pd.DataFrame({"date": dates})
            daily_solves = (
                streak_df.groupby("date")
                .size()
                .reset_index(name="problems_solved")
            )

            last_30_date = datetime.now().date() - timedelta(days=30)
            last_30 = daily_solves[daily_solves["date"] >= last_30_date]

            fig_streak = px.line(
                last_30,
                x="date",
                y="problems_solved",
                title="Problems Solved (Last 30 Days)"
            )

            fig_streak.update_traces(line=dict(width=3))
            st.plotly_chart(fig_streak, use_container_width=True)

            # Current streak calculation
            unique_dates = sorted(daily_solves["date"])
            streak = 1

            for i in range(len(unique_dates) - 1, 0, -1):
                if (unique_dates[i] - unique_dates[i - 1]).days == 1:
                    streak += 1
                else:
                    break

            st.success(f"🔥 Current Streak: {streak} days")

            # =====================================================
            # ================= HEATMAP ===========================
            # =====================================================
            st.markdown("## 🟩 Contribution Heatmap")

            heatmap_df = streak_df.copy()
            heatmap_df["count"] = 1
            heatmap_df["date"] = pd.to_datetime(heatmap_df["date"])

            heatmap_df["week"] = heatmap_df["date"].dt.isocalendar().week
            heatmap_df["weekday"] = heatmap_df["date"].dt.weekday

            heatmap_grouped = (
                heatmap_df.groupby(["week", "weekday"])
                .count()
                .reset_index()
            )

            fig_heatmap = px.density_heatmap(
                heatmap_grouped,
                x="week",
                y="weekday",
                z="count",
                nbinsx=52,
                nbinsy=7,
            )

            fig_heatmap.update_layout(
                yaxis=dict(
                    tickmode="array",
                    tickvals=[0,1,2,3,4,5,6],
                    ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                )
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

        else:
            st.info("No solved problems found.")

        # =====================================================
        # ================= TOPIC ANALYSIS ====================
        # =====================================================
        st.markdown("## 📊 Topic Performance Analysis")

        clean_data = topic_stats.sort_values(by="accuracy")
        bottom_10 = clean_data.head(10)
        top_10 = clean_data.tail(10)
        filtered_data = pd.concat([bottom_10, top_10])

        fig_topics = px.bar(
            filtered_data,
            y="topic",
            x="accuracy",
            orientation="h",
            color="accuracy",
            color_continuous_scale="Blues",
            title="Top & Bottom Performing Topics"
        )

        fig_topics.update_layout(height=600, coloraxis_showscale=False)
        st.plotly_chart(fig_topics, use_container_width=True)

        # =====================================================
        # ================= STRONG / WEAK =====================
        # =====================================================
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔥 Strong Topics")
            st.dataframe(
                strong_topics.sort_values(by="accuracy", ascending=False),
                use_container_width=True
            )

        with col2:
            st.markdown("### ⚠ Weak Topics")
            st.dataframe(
                weak_topics.sort_values(by="accuracy"),
                use_container_width=True
            )

        # =====================================================
        # ================= AI SUGGESTIONS ====================
        # =====================================================
        st.markdown("## 🤖 AI Improvement Suggestions")

        if len(weak_topics) == 0:
            st.success("Excellent balance across topics. Increase difficulty to grow further.")
        else:
            difficulty_mean = int(topic_stats["avg_difficulty"].mean())
            st.warning(f"""
            You have {len(weak_topics)} weak topics.

            📌 Strategy:
            • Solve 5–10 problems per weak topic  
            • Target difficulty: {difficulty_mean - 200} to {difficulty_mean}  
            • Maintain consistency for at least 2 weeks  
            • Track streak growth for measurable improvement
            """)