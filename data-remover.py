import pandas as pd

# Load dataset
df = pd.read_csv("gesture_data.csv", header=None)

# Remove only 'point-left'
df = df[df.iloc[:, -1] != "point-right"]

# Save back
df.to_csv("gesture_data.csv", index=False, header=False)

print("point-left removed successfully")