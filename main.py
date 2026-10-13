import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("students.csv")

print("Dataset loaded successfully!")
print(data.head())

# Input features and target
X = data[["study_hours", "attendance", "previous_marks"]]
y = data["final_marks"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("\nModel training completed!")
print(f"Mean Absolute Error: {mae:.2f}")

# Predict for a new student
new_student = pd.DataFrame(
    [[5, 75, 60]],
    columns=["study_hours", "attendance", "previous_marks"]
)

predicted_marks = model.predict(new_student)[0]
predicted_marks = max(0, min(100, predicted_marks))

print(f"\nStudy hours: 5")
print(f"Attendance: 75%")
print(f"Previous marks: 60")
print(f"Predicted final marks: {predicted_marks:.2f}")