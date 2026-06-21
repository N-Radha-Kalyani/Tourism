
# for data manipulation
import pandas as pd
import os
import joblib

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# for hugging face authentication and upload
from huggingface_hub import HfApi

# Define constants
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/RadhaNK/Tourism/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column
df.drop(columns=['CustomerID'], inplace=True)

# Standardize categorical values
df['Gender'] = (
    df['Gender']
    .str.strip()
    .str.lower()
    .replace({'fe male': 'female'})
)

df['MaritalStatus'] = (
    df['MaritalStatus']
    .str.strip()
    .str.lower()
    .replace({'unmarried': 'single'})
)

# Check missing values before imputation
print("\nMissing Values Before Imputation:")
missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]

if len(missing_before) == 0:
    print("No missing values found.")
else:
    print(missing_before)

    # Numerical columns
    numerical_cols = [
        'Age',
        'CityTier',
        'DurationOfPitch',
        'NumberOfPersonVisiting',
        'NumberOfFollowups',
        'PreferredPropertyStar',
        'NumberOfTrips',
        'Passport',
        'PitchSatisfactionScore',
        'OwnCar',
        'NumberOfChildrenVisiting',
        'MonthlyIncome'
    ]

    for col in numerical_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median_value = df[col].median()
            print(
                f"Imputing {null_count} missing values "
                f"in '{col}' using median ({median_value})"
            )
            df[col] = df[col].fillna(median_value)

    # Categorical columns
    categorical_cols = [
        'TypeofContact',
        'Occupation',
        'Gender',
        'ProductPitched',
        'MaritalStatus',
        'Designation'
    ]

    for col in categorical_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            mode_value = df[col].mode()[0]
            print(
                f"Imputing {null_count} missing values "
                f"in '{col}' using mode ('{mode_value}')"
            )
            df[col] = df[col].fillna(mode_value)

# Check missing values after imputation
print("\nMissing Values After Imputation:")
missing_after = df.isnull().sum()
missing_after = missing_after[missing_after > 0]

if len(missing_after) == 0:
    print("No missing values remaining.")
else:
    print(missing_after)

# Encode categorical variables using LabelEncoder
categorical_cols = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

label_encoders = {}

print("\nLabel Encoding Mappings:")

for col in categorical_cols:
    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le

    mapping = dict(
        zip(
            le.classes_,
            le.transform(le.classes_)
        )
    )

    print(f"{col}: {mapping}")

# Save encoders locally
joblib.dump(label_encoders, "label_encoders.joblib")
print("Label encoders saved.")

# Define target
target = 'ProdTaken'

# Features and target
X = df.drop(columns=[target])
y = df[target]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save processed datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)

ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload files to Hugging Face
files = [
    "Xtrain.csv",
    "Xtest.csv",
    "ytrain.csv",
    "ytest.csv",
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id="RadhaNK/Tourism",
        repo_type="dataset",
    )
api.upload_file(
    path_or_fileobj="label_encoders.joblib",
    path_in_repo="label_encoders.joblib",
    repo_id="RadhaNK/tourism_model",
    repo_type="model"
)


print("\nPreprocessing completed successfully.")
print("Train-test datasets and encoders uploaded to Hugging Face.")
