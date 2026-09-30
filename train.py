import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 1. Load Dataset
data = pd.read_csv("health_data.csv")

# 2. Features and Target
X = data[["Age", "BMI", "BloodPressure"]]
y = data["RiskScore"]

# 3. Train Model
model = LinearRegression()
model.fit(X, y)

# 4. Save Serialized Model
joblib.dump(model, "health_model.pkl")
print("SUCCESS: health_model.pkl generated.")