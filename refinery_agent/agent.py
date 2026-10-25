"""The Refinery Controller AGENT (sense -> think -> act).

SENSE : reads temperature, pressure, flow, impurity
THINK : tries every small action ("what if?"), asks the ML models to predict
        purity / yield / safety for each, and scores them
ACT   : picks the best action for heater, pressure valve and pump
"""
import itertools
import os
import sqlite3
import joblib
import pandas as pd

from train_model import FEATURES, train

if not os.path.exists("models.joblib"):
    train()
MODELS = joblib.load("models.joblib")

# How much each actuator step changes the plant (heater, valve, pump)
HEATER_STEP, VALVE_STEP, PUMP_STEP = 5.0, 0.25, 4.0
PENALTY = {"safe": 0, "warning": 25, "danger": 1000}


def apply_action(state, heater, valve, pump):
    """Predict the sensor values after an action. +1 = more, -1 = less, 0 = keep.
    heater +1 = heat up | valve +1 = close (pressure up) | pump +1 = more flow"""
    return {
        "temperature": state["temperature"] + heater * HEATER_STEP,
        "pressure": state["pressure"] + valve * VALVE_STEP,
        "flow": min(90, max(30, state["flow"] + pump * PUMP_STEP)),
        "impurity": state["impurity"],
    }


def predict(state):
    row = pd.DataFrame([state])[FEATURES]
    return (float(MODELS["purity"].predict(row)[0]),
            float(MODELS["yield"].predict(row)[0]),
            str(MODELS["safety"].predict(row)[0]))


def decide(state):
    """Return the best action and the predictions for the current state."""
    best = None
    for heater, valve, pump in itertools.product([-1, 0, 1], repeat=3):
        future = apply_action(state, heater, valve, pump)
        purity, yld, safety = predict(future)
        effort = abs(heater) + abs(valve) + abs(pump)          # prefer small moves
        score = purity + yld - PENALTY[safety] - 0.5 * effort
        if best is None or score > best["score"]:
            best = {"heater": heater, "valve": valve, "pump": pump, "score": score}
    purity, yld, safety = predict(state)
    best.update(purity=purity, yield_=yld, safety=safety)
    return best


# ---- tiny database log of every decision -------------------------------
def log_decision(state, d):
    with sqlite3.connect("refinery.db") as con:
        con.execute("""CREATE TABLE IF NOT EXISTS agent_log(
            time TEXT DEFAULT CURRENT_TIMESTAMP, temperature REAL, pressure REAL,
            flow REAL, impurity REAL, purity REAL, yield REAL, safety TEXT,
            heater INT, valve INT, pump INT)""")
        con.execute("INSERT INTO agent_log(temperature,pressure,flow,impurity,purity,yield,safety,heater,valve,pump) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (state["temperature"], state["pressure"], state["flow"], state["impurity"],
                     d["purity"], d["yield_"], d["safety"], d["heater"], d["valve"], d["pump"]))
