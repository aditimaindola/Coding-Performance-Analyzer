import requests
import pandas as pd


# -------------------------------
# STEP 1: Fetch Data from API
# -------------------------------
def fetch_data(username):
    url = f"https://codeforces.com/api/user.status?handle={username}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Failed to connect to Codeforces API")

    data = response.json()

    if data["status"] != "OK":
        raise Exception("Invalid username or API error")

    return data["result"]


# -------------------------------
# STEP 2: Build Raw DataFrame
# -------------------------------
def build_dataframe(submissions):
    rows = []

    for sub in submissions:
        problem = sub.get("problem", {})
        verdict = sub.get("verdict", "")

        rating = problem.get("rating", 0)
        tags = problem.get("tags", [])

        # Skip if no rating or tags
        if not tags or rating == 0:
            continue

        topic = tags[0]  # Take first tag as primary topic

        rows.append({
            "topic": topic,
            "difficulty": rating,
            "solved": 1 if verdict == "OK" else 0
        })

    df = pd.DataFrame(rows)
    return df


# -------------------------------
# STEP 3: Compute Topic Metrics
# -------------------------------
def compute_topic_metrics(df):
    topic_stats = df.groupby("topic").agg(
        accuracy=("solved", "mean"),
        avg_difficulty=("difficulty", "mean"),
        total_attempts=("solved", "count"),
        total_solved=("solved", "sum")
    ).reset_index()

    return topic_stats


# -------------------------------
# STEP 4: Add Dynamic Labels
# -------------------------------
def add_labels(topic_stats):
    mean_accuracy = topic_stats["accuracy"].mean()

    topic_stats["label"] = topic_stats["accuracy"].apply(
        lambda x: 1 if x < mean_accuracy else 0
    )

    print("\nOverall Mean Accuracy:", round(mean_accuracy, 4))

    weak_count = topic_stats["label"].sum()
    strong_count = len(topic_stats) - weak_count

    print("Weak Topics:", weak_count)
    print("Strong Topics:", strong_count)

    return topic_stats


# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    username = input("Enter Codeforces username: ")

    print("\nFetching submissions...")
    submissions = fetch_data(username)

    print("Building dataset...")
    df = build_dataframe(submissions)

    print("Computing topic metrics...")
    topic_stats = compute_topic_metrics(df)

    print("Adding labels...")
    topic_stats = add_labels(topic_stats)

    print("\nFinal Topic Analysis:\n")
    print(topic_stats.sort_values(by="accuracy"))