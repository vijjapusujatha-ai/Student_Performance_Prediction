import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

os.makedirs("models", exist_ok=True)
data = pd.read_csv("dataset/student_data.csv")

X = data[["StudyHours","Attendance","PreviousMarks","Assignments"]]
y = data["FinalScore"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "models/model.pkl")

print("Model Saved Successfully!")
