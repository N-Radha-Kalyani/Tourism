import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Visit with Us",
    page_icon="✈️",
    layout="wide"
)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="RadhaNK/tourism_model",
            repo_type="model",
            filename="best_tourism_model_v1.joblib"
        )

        model = joblib.load(model_path)

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


model = load_model()


# -------------------------
# Sidebar Diagnostics
# -------------------------
st.sidebar.header("Diagnostics")

st.sidebar.write("Model Type:", type(model).__name__)

if hasattr(model, "named_steps"):
    st.sidebar.write(
        "Pipeline Steps:",
        list(model.named_steps.keys())
    )

if hasattr(model, "feature_names_in_"):
    st.sidebar.write(
        "Training Features:",
        list(model.feature_names_in_)
    )


# -------------------------
# Header
# -------------------------
st.title("✈️ Visit with Us")

st.subheader(
    "Wellness Tourism Package Purchase Prediction"
)


# -------------------------
# Inputs
# -------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 35)

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
        1,
        20,
        2
    )

    number_of_followups = st.number_input(
        "Number of Follow-ups",
        0,
        20,
        2
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
        "Number of Trips",
        0,
        20,
        2
    )

    passport = st.selectbox(
        "Has Passport",
        [0, 1]
    )

    pitch_satisfaction_score = st.slider(
        "Pitch Satisfaction Score",
        1,
        5,
        3
    )

    own_car = st.selectbox(
        "Owns Car",
        [0, 1]
    )


st.header("Additional Information")

col3, col4 = st.columns(2)

with col3:
    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        0,
        10,
        0
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
        0,
        1000000,
        25000
    )

    duration_of_pitch = st.number_input(
        "Duration Of Pitch",
        1,
        120,
        20
    )

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):

    try:

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

        st.subheader("Input Sent to Model")
        st.dataframe(input_df)

        # Probability prediction
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
            prediction = int(probability >= 0.45)

            st.write("Purchase Probability:", f"{probability:.2%}")
        else:
            prediction = int(model.predict(input_df)[0])
            st.write("Raw Prediction:", prediction)

        st.write("Final Prediction:", prediction)

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "✅ This customer is likely to purchase the Wellness Tourism Package."
            )
        else:
            st.warning(
                "❌ This customer is unlikely to purchase the Wellness Tourism Package."
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

        st.subheader("Input Causing Error")
        st.dataframe(input_df)
