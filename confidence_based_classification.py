import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(path):
    df = pd.read_csv(path, sep=';')
    df['Result'] = (df['G3'] >= 10).astype(int)
    X = df[['studytime', 'failures', 'absences', 'G1', 'G2']]
    y = df['Result']
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate(model, X_test, y_test, threshold):
    probs = model.predict_proba(X_test)[:, 1]
    preds = []
    rejected = 0

    for p in probs:
        if p >= threshold:
            preds.append(1)
        elif p <= 1 - threshold:
            preds.append(0)
        else:
            preds.append(None)
            rejected += 1

    final_preds = []
    final_true = []

    for p, t in zip(preds, y_test):
        if p is not None:
            final_preds.append(p)
            final_true.append(t)

    acc = accuracy_score(final_true, final_preds) if final_preds else 0
    rejection_rate = rejected / len(y_test)

    return acc, rejection_rate


def predict_student(model, threshold):
    print("\nEnter student details")
    studytime = int(input("Study Time (1-4): "))
    failures = int(input("Past Failures: "))
    absences = int(input("Absences: "))
    g1 = int(input("G1 (0-20): "))
    g2 = int(input("G2 (0-20): "))

    data = pd.DataFrame([{
    'studytime': studytime,
    'failures': failures,
    'absences': absences,
    'G1': g1,
    'G2': g2
    }])

    prob = model.predict_proba(data)[0][1]

    print(f"\nProbability of Passing: {prob:.4f}")

    if prob >= threshold:
        print("Decision: PASS")
    elif prob <= 1 - threshold:
        print("Decision: FAIL")
    else:
        print("Decision: REJECTED (Low Confidence)")


if __name__ == "__main__":
    X, y = load_data("student-mat.csv")
    model, X_test, y_test = train_model(X, y)

    threshold = 0.7
    acc, rej = evaluate(model, X_test, y_test, threshold)

    print("\nModel Performance")
    print("Accuracy (confident predictions):", round(acc, 4))
    print("Rejection Rate:", round(rej, 4))

    predict_student(model, threshold)