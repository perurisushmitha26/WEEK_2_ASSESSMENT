# TASK 2: PREPROCESSING, FEATURES AND PIPELINE


# STEP 1: IMPORT LIBRARIES


import os
import numpy as np
import pandas as pd
import joblib


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier




# STEP 2: CREATE FOLDERS


os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)




# STEP 3: CREATE LOAN DATASET


np.random.seed(42)


n = 600


df = pd.DataFrame({


    "application_date": pd.date_range(
        "2024-01-01", periods=n, freq="D"
    ).astype(str),


    "income": np.random.lognormal(10.8, 0.4, n),


    "loan_amount": np.random.lognormal(11.2, 0.5, n),


    "credit_score": np.random.randint(300, 851, n),


    "dependents": np.random.randint(0, 5, n),


    "employment_type": np.random.choice(
        ["Salaried", "Self-employed", "Business"], n
    ),


    "marital_status": np.random.choice(
        ["Single", "Married"], n
    ),


    "education": np.random.choice(
        ["High School", "Graduate", "Post Graduate"], n
    )
})




# STEP 4: CREATE TARGET COLUMN


df["loan_approved"] = (
    (df["credit_score"] >= 650) &
    (df["income"] >= 40000) &
    (df["loan_amount"] <= 150000)
).astype(int)




# STEP 5: INTRODUCE MISSING VALUES


for col in ["income", "loan_amount", "credit_score", "dependents"]:


    df.loc[
        df.sample(frac=0.05, random_state=42).index, col
    ] = np.nan




for col in ["employment_type", "marital_status", "education"]:


    df.loc[
        df.sample(frac=0.05, random_state=10).index, col
    ] = np.nan




df.to_csv("data/loan_data.csv", index=False)


print("Dataset created successfully")


print("\nMissing values:")
print(df.isnull().sum())




# STEP 6: SEPARATE FEATURES AND TARGET


X = df.drop("loan_approved", axis=1)


y = df["loan_approved"]




# STEP 7: TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)




# STEP 8: TRY KNN IMPUTER ON NUMERIC COLUMNS


numeric_columns = [
    "income",
    "loan_amount",
    "credit_score",
    "dependents"
]


knn_imputer = KNNImputer(n_neighbors=5)


knn_result = knn_imputer.fit_transform(
    X_train[numeric_columns]
)


print("\nKNN Imputer Result:")
print("Original shape:", X_train[numeric_columns].shape)
print("After imputation:", knn_result.shape)




# STEP 9: DEFINE COLUMN TYPES


categorical_columns = [
    "employment_type",
    "marital_status"
]


ordinal_columns = ["education"]


date_columns = ["application_date"]




# STEP 10: NUMERIC PIPELINE


numeric_pipeline = Pipeline([


    ("imputer", SimpleImputer(strategy="median")),


    ("scaler", StandardScaler())


])




# STEP 11: CATEGORICAL PIPELINE


categorical_pipeline = Pipeline([


    ("imputer", SimpleImputer(strategy="most_frequent")),


    ("encoder", OneHotEncoder(handle_unknown="ignore"))


])




# STEP 12: ORDINAL PIPELINE


ordinal_pipeline = Pipeline([


    ("imputer", SimpleImputer(strategy="most_frequent")),


    ("encoder", OrdinalEncoder(
        categories=[[
            "High School",
            "Graduate",
            "Post Graduate"
        ]],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))


])




# STEP 13: DATE FEATURE ENGINEERING


def extract_date(X):


    dates = pd.to_datetime(X.iloc[:, 0])


    return np.column_stack([
        dates.dt.year,
        dates.dt.month,
        dates.dt.day
    ])




date_pipeline = Pipeline([


    ("date_features", FunctionTransformer(extract_date)),


    ("imputer", SimpleImputer(strategy="median")),


    ("scaler", StandardScaler())


])




# STEP 14: INTERACTION AND LOG FEATURES


def create_new_features(X):


    income = X[:, 0]


    loan = X[:, 1]


    interaction = income / (loan + 1)


    log_feature = np.log1p(loan)


    return np.column_stack([
        interaction,
        log_feature
    ])




feature_pipeline = Pipeline([


    ("imputer", SimpleImputer(strategy="median")),


    ("feature_engineering",
     FunctionTransformer(create_new_features)),


    ("scaler", StandardScaler())


])




# STEP 15: COLUMN TRANSFORMER


preprocessor = ColumnTransformer([


    ("numeric", numeric_pipeline, numeric_columns),


    ("categorical", categorical_pipeline, categorical_columns),


    ("ordinal", ordinal_pipeline, ordinal_columns),


    ("date", date_pipeline, date_columns),


    ("features", feature_pipeline, ["income", "loan_amount"])


])




# STEP 16: BUILD MACHINE LEARNING PIPELINE


model = Pipeline([


    ("preprocessor", preprocessor),


    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))


])




# STEP 17: 5-FOLD CROSS VALIDATION


cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)


print("\nPipeline CV accuracy:", round(cv_scores.mean(), 2))




# STEP 18: TRAIN MODEL


model.fit(X_train, y_train)


print("\nModel trained successfully")




# STEP 19: SAVE MODEL USING JOBLIB


joblib.dump(model, "models/loan_pipeline.joblib")


print("Pipeline saved successfully")




# STEP 20: LOAD SAVED MODEL


loaded_model = joblib.load("models/loan_pipeline.joblib")


print("Pipeline loaded successfully")




# STEP 21: CREATE 3 NEW RAW ROWS


new_rows = pd.DataFrame({


    "application_date": [
        "2025-01-10",
        "2025-02-15",
        "2025-03-20"
    ],


    "income": [
        30000,
        90000,
        25000
    ],


    "loan_amount": [
        180000,
        50000,
        100000
    ],


    "credit_score": [
        500,
        800,
        450
    ],


    "dependents": [
        2,
        1,
        3
    ],


    "employment_type": [
        "Salaried",
        "Business",
        "Self-employed"
    ],


    "marital_status": [
        "Single",
        "Married",
        "Single"
    ],


    "education": [
        "Graduate",
        "Post Graduate",
        "High School"
    ]


})




# STEP 22: PREDICT 3 NEW ROWS


predictions = loaded_model.predict(new_rows)


print("\nPredictions for 3 new rows:", predictions)


print("\nTASK 2 COMPLETED SUCCESSFULLY!")
