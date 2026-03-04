import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from build_dataset import fetch_data, build_dataframe, compute_topic_metrics, add_labels


def train(username):
    print("Fetching data...")
    submissions = fetch_data(username)

    print("Building dataset...")
    df = build_dataframe(submissions)

    print("Computing topic metrics...")
    topic_stats = compute_topic_metrics(df)

    print("Adding labels...")
    topic_stats = add_labels(topic_stats)

    # Features
    X = topic_stats[["accuracy", "avg_difficulty", "total_attempts"]]
    y = topic_stats["label"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("Training RandomForest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)

    print("\nModel Evaluation:\n")
    print(classification_report(y_test, y_pred))

    # Save Model
    joblib.dump(model, "model/performance_model.pkl")
    print("\nModel saved to model/performance_model.pkl")

    return model


if __name__ == "__main__":
    username = input("Enter Codeforces username: ")
    train(username)