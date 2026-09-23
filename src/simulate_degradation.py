import pandas as pd

FEEDBACK_FILE = "data/feedback.csv"

df = pd.read_csv(FEEDBACK_FILE)

# Force many positive failures to become incorrect predictions.
# This is only for testing the adaptive controller.
failure_rows = df[df["actual"] == 1].index

# Change predictions for failure records to 0
df.loc[failure_rows, "prediction"] = 0

df.to_csv(FEEDBACK_FILE, index=False)

print("Degraded feedback created.")
print("Actual failures:", len(failure_rows))