import pandas as pd

def process_submissions(submissions):
    rows = []

    for sub in submissions:
        problem = sub.get("problem", {})
        verdict = sub.get("verdict", "")

        rating = problem.get("rating", 0)
        tags = problem.get("tags", [])

        if not tags:
            continue

        topic = tags[0]  # take first tag

        rows.append({
            "topic": topic,
            "difficulty": rating,
            "solved": 1 if verdict == "OK" else 0
        })

    df = pd.DataFrame(rows)

    # Group by topic
    topic_stats = df.groupby("topic").agg({
        "solved": "mean",
        "difficulty": "mean"
    }).reset_index()

    topic_stats.rename(columns={"solved": "accuracy"}, inplace=True)

    return topic_stats