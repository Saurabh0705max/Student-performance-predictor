# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error

# st.set_page_config(
#     page_title="Student Performance Predictor",
#     page_icon="🎓",
#     layout="centered"
# )

# @st.cache_resource
# def train_model():
#     data = pd.read_csv("students.csv")

#     X = data[["study_hours", "attendance", "previous_marks"]]
#     y = data["final_marks"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=42
#     )

#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     predictions = model.predict(X_test)
#     mae = mean_absolute_error(y_test, predictions)

#     return model, mae

# st.title("🎓 Student Performance Predictor")
# st.write(
#     "Enter student details to estimate the final marks "
#     "using a Machine Learning model."
# )

# try:
#     model, mae = train_model()

#     st.subheader("Enter Student Details")

#     study_hours = st.number_input(
#         "Daily study hours",
#         min_value=0.0,
#         max_value=24.0,
#         value=5.0,
#         step=0.5
#     )

#     attendance = st.slider(
#         "Attendance (%)",
#         min_value=0,
#         max_value=100,
#         value=75
#     )

#     previous_marks = st.number_input(
#         "Previous marks (out of 100)",
#         min_value=0,
#         max_value=100,
#         value=60
#     )

#     if st.button("Predict Marks", type="primary"):
#         new_student = pd.DataFrame(
#             [[study_hours, attendance, previous_marks]],
#             columns=[
#                 "study_hours",
#                 "attendance",
#                 "previous_marks"
#             ]
#         )

#         predicted_marks = model.predict(new_student)[0]
#         predicted_marks = max(0, min(100, predicted_marks))

#         st.success(
#             f"Predicted Final Marks: {predicted_marks:.2f} / 100"
#         )

#         st.info(
#             "This is an educational estimate based on a small "
#             "synthetic dataset, not a reliable real-world forecast."
#         )

#     with st.expander("Model information"):
#         st.write("Algorithm: Linear Regression")
#         st.write(f"Test-set MAE: {mae:.2f} marks")
#         st.write("Dataset: students.csv")

# except FileNotFoundError:
#     st.error(
#         "students.csv not found. Keep it in the same "
#         "folder as app.py."
#     )
# except Exception as e:
#     st.error(f"Something went wrong: {e}")





import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)


# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1 {
    color: #173b67;
}
h2, h3 {
    color: #24527a;
}
div[data-testid="stMetric"] {
    background-color: #ffffff;
    color: #173b67;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #173b67 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL TRAINING ----------------

@st.cache_data
def load_data():
    data = pd.read_csv("students.csv")

    required_columns = [
        "study_hours",
        "attendance",
        "previous_marks",
        "final_marks"
    ]

    if not all(col in data.columns for col in required_columns):
        raise ValueError(
            "students.csv must contain: "
            + ", ".join(required_columns)
        )

    data = data[required_columns].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(data) < 8:
        raise ValueError(
            "At least 8 valid student records are required."
        )

    if not data["final_marks"].between(0, 100).all():
        raise ValueError(
            "Final marks must be between 0 and 100."
        )

    if not data["previous_marks"].between(0, 100).all():
        raise ValueError(
            "Previous marks must be between 0 and 100."
        )

    if not data["attendance"].between(0, 100).all():
        raise ValueError(
            "Attendance must be between 0 and 100."
        )

    if not data["study_hours"].between(0, 24).all():
        raise ValueError(
            "Study hours must be between 0 and 24."
        )

    return data


@st.cache_resource
def train_models():
    data = load_data()

    features = [
        "study_hours",
        "attendance",
        "previous_marks"
    ]

    X = data[features]
    y = data["final_marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42
        )
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results[name] = {
            "MAE": mean_absolute_error(y_test, predictions),
            "R2": r2_score(y_test, predictions)
        }

        trained_models[name] = model

    return data, trained_models, results


# ---------------- HEADER ----------------

st.title("🎓 Student Performance Predictor")
st.write(
    "An educational Machine Learning dashboard "
    "to estimate student performance."
)

st.caption(
    "Project version 2.0 | Built with Python, "
    "Streamlit and Scikit-learn"
)

st.divider()


# ---------------- MAIN APPLICATION ----------------

try:
    data, models, results = train_models()

    # Sidebar
    st.sidebar.title("⚙️ Prediction Settings")

    model_name = st.sidebar.selectbox(
        "Choose ML Model",
        list(models.keys())
    )

    st.sidebar.info(
        "Enter the student's details to estimate "
        "their final marks."
    )

    st.sidebar.caption(
        "Predictions are educational estimates, "
        "not guaranteed results."
    )

    # Dashboard metrics
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Student Records", len(data))

    with col2:
        st.metric(
            "Average Final Marks",
            f"{data['final_marks'].mean():.1f}/100"
        )

    with col3:
        st.metric(
            "Average Attendance",
            f"{data['attendance'].mean():.1f}%"
        )

    st.divider()

    # Prediction form
    st.subheader("🎯 Predict Student Performance")

    left, right = st.columns(2)

    with left:
        study_hours = st.number_input(
            "Daily Study Hours",
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

    with right:
        previous_marks = st.number_input(
            "Previous Marks (out of 100)",
            min_value=0,
            max_value=100,
            value=60
        )

        st.write("Selected model:", model_name)

    if st.button(
        "🚀 Predict Final Marks",
        type="primary",
        use_container_width=True
    ):
        # Check whether inputs are outside training ranges
        inputs = {
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks
        }

        out_of_range = []

        for feature, value in inputs.items():
            minimum = data[feature].min()
            maximum = data[feature].max()

            if value < minimum or value > maximum:
                out_of_range.append(
                    f"{feature.replace('_', ' ').title()}: "
                    f"training range is {minimum:g}–{maximum:g}"
                )

        if out_of_range:
            st.warning(
                "One or more inputs are outside the range "
                "of the training dataset. The prediction "
                "may be unreliable."
            )

            for item in out_of_range:
                st.write(f"- {item}")

        student = pd.DataFrame(
            [[study_hours, attendance, previous_marks]],
            columns=[
                "study_hours",
                "attendance",
                "previous_marks"
            ]
        )

        model = models[model_name]
        prediction = float(model.predict(student)[0])

        # Display a bounded score, but retain the original
        # prediction for warning purposes.
        displayed_prediction = max(
            0.0, min(100.0, prediction)
        )

        st.divider()
        st.subheader("📈 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.markdown(
            f"""
            <div style="
            background-color: #ffffff;
            color: #173b67;
            padding: 25px;
            border-radius: 12px;
            border: 2px solid #24527a;
            text-align: center;
        ">
            <h3 style="color: #24527a;">
                Predicted Final Marks
            </h3>
            <h1 style="color: #173b67; font-size: 42px;">
                {displayed_prediction:.1f}/100
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

        st.progress(displayed_prediction / 100)

        with result_col2:
            if displayed_prediction >= 80:
                category = "Excellent 🌟"
                suggestion = (
                    "Maintain consistent study habits "
                    "and continue practising."
                )
            elif displayed_prediction >= 60:
                category = "Good 👍"
                suggestion = (
                    "Revise difficult topics and practise "
                    "sample questions regularly."
                )
            elif displayed_prediction >= 40:
                category = "Needs Improvement 📚"
                suggestion = (
                    "Create a study timetable and focus "
                    "on topics where you lose marks."
                )
            else:
                category = "Extra Support Recommended 💡"
                suggestion = (
                    "Start with fundamental concepts, "
                    "practise daily and ask for help "
                    "when needed."
                )

            st.markdown(f"### {category}")
            st.write(suggestion)

        st.caption(
            "Performance categories are illustrative thresholds, "
            "not official grading standards."
        )

        st.info(
            "This estimate is based on a small synthetic dataset. "
            "It should not be used to make academic decisions."
        )

    st.divider()

    # Visualizations
    st.subheader("📊 Explore Student Data")

    graph_type = st.selectbox(
        "Choose a visualization",
        [
            "Study Hours vs Final Marks",
            "Attendance vs Final Marks",
            "Previous Marks vs Final Marks",
            "Marks Distribution"
        ]
    )

    if graph_type == "Study Hours vs Final Marks":
        fig = px.scatter(
            data,
            x="study_hours",
            y="final_marks",
            title="Study Hours vs Final Marks",
            labels={
                "study_hours": "Daily Study Hours",
                "final_marks": "Final Marks"
            },
            hover_data=["attendance", "previous_marks"]
        )

    elif graph_type == "Attendance vs Final Marks":
        fig = px.scatter(
            data,
            x="attendance",
            y="final_marks",
            title="Attendance vs Final Marks",
            labels={
                "attendance": "Attendance (%)",
                "final_marks": "Final Marks"
            },
            hover_data=["study_hours", "previous_marks"]
        )

    elif graph_type == "Previous Marks vs Final Marks":
        fig = px.scatter(
            data,
            x="previous_marks",
            y="final_marks",
            title="Previous Marks vs Final Marks",
            labels={
                "previous_marks": "Previous Marks",
                "final_marks": "Final Marks"
            },
            hover_data=["study_hours", "attendance"]
        )

    else:
        fig = px.histogram(
            data,
            x="final_marks",
            nbins=10,
            title="Distribution of Final Marks",
            labels={"final_marks": "Final Marks"}
        )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Model evaluation
    st.subheader("🤖 Machine Learning Model Comparison")

    comparison = pd.DataFrame(results).T.reset_index()
    comparison.columns = ["Model", "MAE", "R2"]

    st.dataframe(
        comparison.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "MAE: average absolute prediction error; lower is better. "
        "R²: fit score; it can be negative on test data, and "
        "a higher score is not necessarily reliable on a tiny dataset."
    )

    with st.expander("🔍 View Dataset"):
        st.dataframe(data, use_container_width=True)

    with st.expander("ℹ️ About This Project"):
        st.write("""
        **Goal:** Estimate final student marks using three inputs.

        **Input features:**
        - Daily study hours
        - Attendance percentage
        - Previous marks

        **Target:** Final marks out of 100.

        **Algorithms:** Linear Regression and Random Forest.

        **Limitations:** The current dataset is small and synthetic.
        Better evaluation requires more representative real data.
        """)

except FileNotFoundError:
    st.error(
        "students.csv was not found. Keep it in the same "
        "folder as app.py."
    )

except Exception as error:
    st.error(f"Something went wrong: {error}")