"""Step 1 - Build the DATABASE (training data).

There is no free public "refinery" dataset that matches our sensors, so we
create a synthetic one from simple process-engineering rules + noise.
It is saved in two places:
  - refinery_data.csv   (open it in Excel)
  - refinery.db         (SQLite database, table: training_data)
"""
import sqlite3
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

# SENSOR READINGS (what the agent can "see")
temperature = rng.uniform(290, 410, N)   # deg C
pressure = rng.uniform(1.0, 5.5, N)      # bar
flow = rng.uniform(30, 90, N)            # feed flow, %
impurity = rng.uniform(0, 100, N)        # chemical sensor, ppm

# TRUE PROCESS RULES (hidden from the agent - it must LEARN them)
purity = (99 - 0.004 * (temperature - 350) ** 2 - 1.5 * (pressure - 3) ** 2
          - 0.03 * impurity + rng.normal(0, 0.4, N))
yield_ = (95 - 0.003 * (temperature - 345) ** 2 - 0.8 * (pressure - 3.2) ** 2
          - 0.01 * (flow - 60) ** 2 + rng.normal(0, 0.5, N))
purity = np.clip(purity, 40, 99.9)
yield_ = np.clip(yield_, 30, 99)

# SAFETY LABEL
risk = np.maximum.reduce([(temperature - 350) / 40, (pressure - 3) / 1.5,
                          (impurity - 50) / 50]) + rng.normal(0, 0.03, N)
safety = np.where(risk > 1.0, "danger", np.where(risk > 0.65, "warning", "safe"))

df = pd.DataFrame({
    "temperature": temperature.round(1), "pressure": pressure.round(2),
    "flow": flow.round(1), "impurity": impurity.round(1),
    "purity": purity.round(2), "yield": yield_.round(2), "safety": safety,
})

df.to_csv("refinery_data.csv", index=False)
with sqlite3.connect("refinery.db") as con:
    df.to_sql("training_data", con, if_exists="replace", index=False)

print(f"Saved {len(df)} rows to refinery_data.csv and refinery.db")
print(df["safety"].value_counts().to_string())
