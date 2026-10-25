"""Step 3 - The website. Run:  python app.py   then open http://127.0.0.1:5000"""
import random
from flask import Flask, jsonify, render_template

import agent

app = Flask(__name__)

START = {"temperature": 372.0, "pressure": 3.8, "flow": 52.0, "impurity": 45.0}
plant = dict(START)
history = []


def sensors():
    return {k: round(v, 2) for k, v in plant.items()}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/state")
def state():
    return jsonify(sensors=sensors(), history=history[-40:])


@app.route("/api/step", methods=["POST"])
def step():
    """One agent cycle: sense -> think -> act, then the plant drifts a bit."""
    before = sensors()
    d = agent.decide(plant)
    new = agent.apply_action(plant, d["heater"], d["valve"], d["pump"])
    # random real-world disturbance
    new["temperature"] += random.uniform(-2, 2)
    new["pressure"] += random.uniform(-0.1, 0.1)
    new["impurity"] = min(100, max(0, new["impurity"] + random.uniform(-3, 3)))
    plant.update(new)
    agent.log_decision(before, d)
    history.append({**before, "purity": d["purity"], "yield": d["yield_"], "safety": d["safety"],
                    "heater": d["heater"], "valve": d["valve"], "pump": d["pump"]})
    return jsonify(sensors=sensors(), history=history[-40:])


@app.route("/api/disturb", methods=["POST"])
def disturb():
    """Simulate a fault: temperature and pressure spike."""
    plant["temperature"] += 25
    plant["pressure"] += 0.8
    plant["impurity"] = min(100, plant["impurity"] + 15)
    return jsonify(sensors=sensors(), history=history[-40:])


@app.route("/api/reset", methods=["POST"])
def reset():
    plant.update(START)
    history.clear()
    return jsonify(sensors=sensors(), history=[])


if __name__ == "__main__":
    app.run(debug=False)
