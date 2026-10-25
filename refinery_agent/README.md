# Refinery Controller – Simple AI Agent (Python)

An AI agent that controls an oil refinery process. It reads sensors,
uses Machine Learning to predict what will happen, and chooses actions on
heaters, valves and pumps. It has a website-style interface.

---

## 1. How to run (3 steps)

```bash
pip install -r requirements.txt     # 1. install libraries (once)
python generate_data.py             # 2. create the database + train (see note)
python train_model.py               #    train the ML models
python app.py                       # 3. start the website
```

Open **http://127.0.0.1:5000** in your browser.

> Python 3.9+ is needed. Steps 2 only need to be done once.

### How to use the website
| Button | What it does |
|---|---|
| ▶ Run 1 step | The agent senses, thinks and acts one time |
| ⏵ Auto-run | Runs a step every second |
| ⚠ Cause a fault | Spikes temperature/pressure - watch the agent fix it |
| ↺ Reset | Starts again |

**Good demo:** press Auto-run, wait until everything is *safe*, then press
*Cause a fault*. Safety turns *warning/danger* and the agent cools the plant
down step by step.

---

## 2. Files

| File | Purpose |
|---|---|
| `generate_data.py` | Creates the **database** (`refinery.db` + `refinery_data.csv`) |
| `train_model.py` | Trains the **ML models** and saves `models.joblib` |
| `agent.py` | The **agent** (sense → think → act) |
| `app.py` | The website server (Flask) |
| `templates/index.html` | The web page |

---

## 3. PEAS chart

| Agent Type | Performance Measure | Environment | Actuators | Sensors |
|---|---|---|---|---|
| Refinery controller | Purity, yield, safety | Refinery, operators | Valves, pumps, heaters, displays | Temperature, pressure, chemical sensors |

### How each PEAS part is used in the code

| PEAS | In this project | Where |
|---|---|---|
| **Performance** | Score = predicted **purity + yield − safety penalty** (danger = −1000, warning = −25). Highest score wins. | `agent.py` → `decide()` |
| **Environment** | A simulated refinery whose values drift randomly; "Cause a fault" simulates an accident. Operators = the person using the website. | `app.py` (`plant`, `step()`, `disturb()`) |
| **Actuators** | **Heater** (heat up/down), **pressure valve** (close/open), **pump** (faster/slower). **Displays** = the web page. | `agent.py` → `apply_action()`, `templates/index.html` |
| **Sensors** | **Temperature**, **pressure**, **flow**, and a **chemical sensor** (impurity ppm). | `START` dict in `app.py`; `FEATURES` in `train_model.py` |

---

## 4. The database

There is no free public dataset with exactly these refinery sensors, so the
database is **generated** by `generate_data.py` using simple process rules
plus random noise (5000 rows). It is stored in:

* `refinery_data.csv` – open in Excel
* `refinery.db` – SQLite database
  * table `training_data` – the 5000 rows the ML learns from
  * table `agent_log` – every decision the agent makes (created automatically)

| Column | Meaning |
|---|---|
| temperature, pressure, flow, impurity | sensor inputs |
| purity, yield | quality results (numbers) |
| safety | label: `safe` / `warning` / `danger` |

View it with free tool *DB Browser for SQLite*, or:
`python -c "import sqlite3;print(sqlite3.connect('refinery.db').execute('select * from agent_log').fetchall())"`

*(If your teacher wants real data, you can replace the CSV with a real
process dataset, e.g. the "Debutanizer column" or "Sulfur recovery unit"
datasets used in soft-sensor research, keeping the same column names.)*

---

## 5. Machine Learning – which, why, where

**Algorithm: Random Forest** (from scikit-learn) – 3 models:

| Model | Type | Predicts | Accuracy on test data |
|---|---|---|---|
| Purity model | Random Forest **Regressor** | purity % | R² ≈ 0.99 |
| Yield model | Random Forest **Regressor** | yield % | R² ≈ 0.97 |
| Safety model | Random Forest **Classifier** | safe / warning / danger | ≈ 97 % |

**Why Random Forest?**
* Easy to understand: many decision trees vote together.
* Works well on small tabular sensor data, no need to scale numbers.
* Handles curved relationships (purity is best near ~350 °C and drops on both sides).
* Does both jobs we need: predicting numbers (regression) and categories (classification).
* Hard to overfit and trains in seconds.

**Where is it used?**
1. **Trained** in `train_model.py` on the table `training_data`.
2. **Used** in `agent.py`:
   * `predict()` asks the 3 models for purity, yield and safety.
   * `decide()` tries all 27 small actions (heater −1/0/+1 × valve −1/0/+1 × pump −1/0/+1),
     asks the ML "what would happen if I did this?", and picks the action with the best score.

This type of agent is called a **model-based, utility-based agent**:
it has a model of the world (the ML), and chooses the action that gives the
highest utility (best purity + yield, while staying safe).

### Agent loop (each "step")
```
SENSE  -> read temperature, pressure, flow, impurity
THINK  -> for each of 27 actions: ML predicts purity, yield, safety -> score
ACT    -> apply best heater / valve / pump action, show on display, save to agent_log
```

---

## 6. Limits (be honest in your presentation)
* The plant is a **simulation**, not a real refinery.
* The data is **synthetic** (generated from rules), so the ML re-learns those rules.
* A real refinery controller would add many more sensors, hard safety interlocks and human approval.
