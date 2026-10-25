"""Step 2 - Train the Machine Learning models (Random Forest).

Input  : temperature, pressure, flow, impurity   (sensor values)
Output : purity (number), yield (number), safety (safe/warning/danger)
"""
import sqlite3
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

FEATURES = ["temperature", "pressure", "flow", "impurity"]


def train():
    with sqlite3.connect("refinery.db") as con:
        df = pd.read_sql("SELECT * FROM training_data", con)

    X = df[FEATURES]
    X_tr, X_te, idx_tr, idx_te = train_test_split(X, df.index, test_size=0.2, random_state=1)

    models = {}
    for target in ["purity", "yield"]:
        m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, random_state=1, n_jobs=-1)
        m.fit(X_tr, df.loc[idx_tr, target])
        print(f"{target:7s} model  R2 score = {r2_score(df.loc[idx_te, target], m.predict(X_te)):.3f}")
        models[target] = m

    clf = RandomForestClassifier(n_estimators=40, min_samples_leaf=5, random_state=1, n_jobs=-1)
    clf.fit(X_tr, df.loc[idx_tr, "safety"])
    print(f"safety  model  accuracy = {accuracy_score(df.loc[idx_te, 'safety'], clf.predict(X_te)):.3f}")
    models["safety"] = clf

    joblib.dump(models, "models.joblib", compress=3)
    print("Saved models.joblib")


if __name__ == "__main__":
    train()
