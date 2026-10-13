import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def train_model():
    data = pd.read_csv("students.csv")

    X = data[["study_hours", "attendance", "previous_marks"]]
    y = data["final_marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    return model, mae

st.title("🎓 Student Performance Predictor")
st.write(
    "Enter student details to estimate the final marks "
    "using a Machine Learning model."
)

try:
    model, mae = train_model()

    st.subheader("Enter Student Details")

    study_hours = st.number_input(
        "Daily study hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    attendance = st.slider(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=75
    )

    previous_marks = st.number_input(
        "Previous marks (out of 100)",
        min_value=0,
        max_value=100,
        value=60
    )

    if st.button("Predict Marks", type="primary"):
        new_student = pd.DataFrame(
            [[study_hours, attendance, previous_marks]],
            columns=[
                "study_hours",
                "attendance",
                "previous_marks"
            ]
        )

        predicted_marks = model.predict(new_student)[0]
        predicted_marks = max(0, min(100, predicted_marks))

        st.success(
            f"Predicted Final Marks: {predicted_marks:.2f} / 100"
        )

        st.info(
            "This is an educational estimate based on a small "
            "synthetic dataset, not a reliable real-world forecast."
        )

    with st.expander("Model information"):
        st.write("Algorithm: Linear Regression")
        st.write(f"Test-set MAE: {mae:.2f} marks")
        st.write("Dataset: students.csv")

except FileNotFoundError:
    st.error(
        "students.csv not found. Keep it in the same "
        "folder as app.py."
    )
except Exception as e:
    st.error(f"Something went wrong: {e}")