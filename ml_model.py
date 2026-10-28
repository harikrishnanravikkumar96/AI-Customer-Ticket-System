import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "sample_tickets.csv"
MODEL_PATH = BASE_DIR / "solution_model.joblib"

TRAINING_DATA = [
    # Password recovery
    ("forgot my password", "password_reset"),
    ("cannot remember my login password", "password_reset"),
    ("need help recovering my account password", "password_reset"),
    ("password reset link is needed", "password_reset"),

    # Payment issues
    ("my payment failed", "payment_check"),
    ("money was deducted but payment failed", "payment_check"),
    ("payment transaction is unsuccessful", "payment_check"),
    ("my card payment did not complete", "payment_check"),

    # Order tracking
    ("where is my order", "order_tracking"),
    ("my order has not arrived", "order_tracking"),
    ("I want to track my delivery", "order_tracking"),
    ("my package is delayed", "order_tracking"),

    # Login access
    ("I cannot log in to my account", "login_troubleshooting"),
    ("unable to access my account", "login_troubleshooting"),
    ("login page is not allowing me to sign in", "login_troubleshooting"),
    ("my account login is not working", "login_troubleshooting"),

    # Technical issues
    ("the application keeps crashing", "technical_support"),
    ("the website shows an error", "technical_support"),
    ("the application is not working properly", "technical_support"),
    ("I get an error when opening the application", "technical_support"),
    ("my laptop keeps restarting", "technical_support"),
    ("laptop restarting issue", "technical_support"),
    ("my computer restarts automatically", "technical_support"),
    ("computer keeps restarting while working", "technical_support"),
    ("my laptop keeps shutting down unexpectedly", "technical_support"),
    ("my computer freezes and restarts", "technical_support"),
    ("laptop is overheating and restarting", "technical_support"),
    ("my laptop is overheating", "technical_support"),
    ("my computer has a heating problem", "technical_support"),
    ("my system keeps crashing", "technical_support"),
    ("computer has a technical problem", "technical_support"),
    ("my laptop is not working properly", "technical_support"),
    ("windows computer restarts repeatedly", "technical_support"),
]

SOLUTIONS = {
    "password_reset":
        "Reset the password using the account recovery process.",

    "payment_check":
        "Check the payment transaction status and verify the payment outcome.",

    "order_tracking":
        "Check the order tracking information and current delivery status.",

    "login_troubleshooting":
        "Verify the account credentials and investigate the login issue.",

    "technical_support":
        "Investigate the reported technical error and follow the troubleshooting process.",
}


def create_dataset():
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ticket_description", "solution_category"])
        writer.writerows(TRAINING_DATA)

    print(f"Training dataset saved to: {DATASET_PATH}")


def train_model():
    texts = [text for text, label in TRAINING_DATA]
    labels = [label for text, label in TRAINING_DATA]

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    model.fit(texts, labels)

    joblib.dump(model, MODEL_PATH)
    print(f"ML model trained and saved to: {MODEL_PATH}")


def predict_solution(subject, description):
    model = joblib.load(MODEL_PATH)

    text = f"{subject}. {description}"
    predicted_category = model.predict([text])[0]

    return SOLUTIONS[predicted_category]


if __name__ == "__main__":
    create_dataset()
    train_model()

    examples = [
        (
            "Unable to log in",
            "I forgot my password and cannot access my account."
        ),
        (
            "Payment failed",
            "My bank account was charged but the transaction failed."
        ),
        (
            "Order delayed",
            "My package has not arrived yet."
        ),
        (
            "Restart Issue",
            "Laptop Restarting Issue."
        ),
        (
            "Computer Problem",
            "My computer restarts automatically while working."
        ),
        (
            "Laptop Overheating",
            "My laptop gets hot and restarts unexpectedly."
        ),
    ]

    print("\nPrediction examples:")

    for subject, description in examples:
        solution = predict_solution(subject, description)
        print(f"\nTicket: {subject}")
        print(f"Predicted solution: {solution}")