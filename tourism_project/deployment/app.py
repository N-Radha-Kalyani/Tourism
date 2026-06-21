import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Visit with Us - Wellness Package Predictor",
    page_icon="✈️",
    layout="wide"
)

# -----------------------------
# Load Model and Encoders
# -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        model_path = hf_hub_download(
            repo_id="RadhaNK/tourism_model",
            repo_type="model",
            filename="best_tourism_model_v1.joblib"
        )

        encoder_path = hf_hub_download(
            repo_id="RadhaNK/tourism_model",
            repo_type="model",
            filename="label_encoders.joblib"
        )

        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)

        return model, encoders

    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()


model, encoders = load_artifacts()

# -----------------------------
# Sidebar Diagnostics
# -----------------------------
st.sidebar.header("Model Diagnostics")

st.sidebar.write("Model Type:", type(model).__name__)

if hasattr(model, "classes_"):
    st.sidebar.write("Classes:", model.classes_)

if hasattr(model, "n_features_in_"):
    st.sidebar.write("Expected Features:", model.n_features_in_)

# -----------------------------
# App Header
# -----------------------------
st.title("✈️ Visit with Us")
st.subheader("Wellness Tourism Package Purchase Prediction")

st.markdown("""
This application predicts whether a customer is likely to purchase the newly introduced
**Wellness Tourism Package**.

Enter the customer details below and click **Predict** to identify potential buyers.
""")

# -----------------------------
# Customer Details
# -----------------------------
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)

    typeof_contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Free Lancer", "Small Business"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2
    )

    number_of_followups = st.number_input(
        "Number of Follow-ups",
        min_value=0,
        max_value=20,
        value=2
    )

with col2:
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        [1, 2, 3, 4, 5]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced", "Unmarried"]
    )

    number_of_trips = st.number_input(
        "Number of Trips per Year",
        min_value=0,
        max_value=20,
        value=2
    )

    passport = st.selectbox(
        "Has Passport",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    pitch_satisfaction_score = st.slider(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    own_car = st.selectbox(
        "Owns Car",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# -----------------------------
# Additional Details
# -----------------------------
st.header("Additional Information")

col3, col4 = st.columns(2)

with col3:
    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=0
    )

    designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

with col4:
    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        max_value=1000000,
        value=25000
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=1,
        max_value=120,
        value=20
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Purchase Probability"):

    try:
        # Create input DataFrame
        input_df = pd.DataFrame([{
            "Age": age,
            "TypeofContact": typeof_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": number_of_person_visiting,
            "NumberOfFollowups": number_of_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_of_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_satisfaction_score,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": number_of_children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income
        }])

        # Match training preprocessing
        input_df["Gender"] = (
            input_df["Gender"]
            .str.strip()
            .str.lower()
        )

        input_df["MaritalStatus"] = (
            input_df["MaritalStatus"]
            .str.strip()
            .str.lower()
            .replace({"unmarried": "single"})
        )

        # Debug Logs
        st.subheader("Debug Logs")

        st.write("Raw Input:")
        st.dataframe(input_df)

        # Apply Label Encoding
        for col, encoder in encoders.items():

            st.write(f"Encoding {col}")

            st.write(
                f"Before: {input_df[col].iloc[0]}"
            )

            input_df[col] = encoder.transform(
                input_df[col]
            )

            st.write(
                f"After: {input_df[col].iloc[0]}"
            )

        st.write("Encoded Input:")
        st.dataframe(input_df)

        # Predict
        prediction = model.predict(input_df)[0]

        st.write("Predicted Class:", int(prediction))

        probability = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]

            probability = probs[1]

            st.write("Prediction Probabilities:")

            st.write({
                "No Purchase (0)": float(probs[0]),
                "Purchase (1)": float(probs[1])
            })

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "✅ This customer is likely to purchase the Wellness Tourism Package."
            )
        else:
            st.warning(
                "❌ This customer is unlikely to purchase the Wellness Tourism Package."
            )

        if probability is not None:
            st.metric(
                "Purchase Probability",
                f"{probability:.2%}"
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

        st.write("Input causing error:")
        st.dataframe(input_df)
