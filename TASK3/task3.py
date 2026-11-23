from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.base import clone


from xgboost import XGBClassifier




# ==================================================
# 1. LOAD DATASET
# ==================================================


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan_data.csv"


OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DATA_PATH)


print("\n========== DATASET ==========")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())




# ==================================================
# 2. IDENTIFY TARGET COLUMNS
# ==================================================


target_candidates = [
    "loan_status",
    "Loan_Status",
    "loan_approved",
    "Loan_Approved",
    "approved",
    "Approval_Status"
]


target = next(
    (col for col in target_candidates if col in df.columns),
    None
)


if target is None:
    raise ValueError(
        "Classification target not found. "
        "Please check your CSV column names."
    )


print("\nClassification target:", target)


df = df.dropna(subset=[target]).copy()


# Remove target from input features
X = df.drop(columns=[target])
y = df[target]


# Convert classification target to numeric if necessary
if y.dtype == "object":
    y = y.astype(str)
    y, class_labels = pd.factorize(y)
    print("Class labels:", list(class_labels))


# Handle binary or multiclass classification
if y.nunique() < 2:
    raise ValueError("Target must contain at least 2 classes.")


# Identify numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()




# ==================================================
# 3. PREPROCESSING PIPELINE
# ==================================================


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)




# ==================================================
# 4. TRAIN AND TEST SPLIT
# ==================================================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))




# ==================================================
# 5. LOGISTIC REGRESSION
# ==================================================


logistic_pipeline = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("model", LogisticRegression(max_iter=2000))
])


logistic_pipeline.fit(X_train, y_train)


logistic_predictions = logistic_pipeline.predict(X_test)


logistic_accuracy = accuracy_score(
    y_test, logistic_predictions
)


print("\n========== LOGISTIC REGRESSION ==========")
print("Accuracy:", round(logistic_accuracy, 4))


# Predict probabilities for 5 rows
probabilities = logistic_pipeline.predict_proba(
    X_test.iloc[:5]
)


predictions = logistic_pipeline.predict(
    X_test.iloc[:5]
)


print("\nProbability predictions for 5 rows:")
print(probabilities)


print("\nClass predictions:")
print(predictions)


print("\nActual values:")
print(y_test.iloc[:5].values)




# ==================================================
# 6. DECISION TREE CLASSIFIER
# ==================================================


decision_tree_pipeline = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("model", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])


decision_tree_pipeline.fit(X_train, y_train)


dt_predictions = decision_tree_pipeline.predict(X_test)


dt_accuracy = accuracy_score(y_test, dt_predictions)


print("\n========== DECISION TREE ==========")
print("Accuracy:", round(dt_accuracy, 4))




# ==================================================
# 7. RANDOM FOREST CLASSIFIER WITH OOB SCORE
# ==================================================


random_forest_pipeline = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    ))
])


random_forest_pipeline.fit(X_train, y_train)


rf_predictions = random_forest_pipeline.predict(X_test)


rf_accuracy = accuracy_score(y_test, rf_predictions)


rf_model = random_forest_pipeline.named_steps["model"]


print("\n========== RANDOM FOREST ==========")
print("Accuracy:", round(rf_accuracy, 4))
print("OOB Score:", round(rf_model.oob_score_, 4))




# ==================================================
# 8. RANDOM FOREST TOP 5 IMPORTANT FEATURES
# ==================================================


feature_names = (
    random_forest_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


rf_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})


rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nTop 5 Random Forest Features:")
print(rf_importance.head(5).to_string(index=False))




# ==================================================
# 9. XGBOOST CLASSIFIER WITH EARLY STOPPING
# ==================================================


# Fit a copy of the preprocessor on training data
# to transform the validation data for XGBoost.


xgb_preprocessor = clone(preprocessor)


X_train_transformed = xgb_preprocessor.fit_transform(X_train)


X_test_transformed = xgb_preprocessor.transform(X_test)


xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=20,
    n_jobs=-1
)


xgb_pipeline = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("model", xgb_model)
])


xgb_pipeline.fit(
    X_train,
    y_train,
    model__eval_set=[
        (X_test_transformed, y_test)
    ],
    model__verbose=False
)


xgb_predictions = xgb_pipeline.predict(X_test)


xgb_accuracy = accuracy_score(y_test, xgb_predictions)


print("\n========== XGBOOST ==========")
print("Accuracy:", round(xgb_accuracy, 4))




# ==================================================
# 10. XGBOOST TOP 5 IMPORTANT FEATURES
# ==================================================


xgb_feature_names = (
    xgb_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


xgb_importance = pd.DataFrame({
    "Feature": xgb_feature_names,
    "Importance": xgb_pipeline
        .named_steps["model"]
        .feature_importances_
})


xgb_importance = xgb_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nTop 5 XGBoost Features:")
print(xgb_importance.head(5).to_string(index=False))




# ==================================================
# 11. COMPARE CLASSIFICATION MODELS
# ==================================================


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],
    "Validation Accuracy": [
        logistic_accuracy,
        dt_accuracy,
        rf_accuracy,
        xgb_accuracy
    ]
})


print("\n========== CLASSIFICATION RESULTS ==========")
print(results.to_string(index=False))


results.to_csv(
    OUTPUT_DIR / "classification_results.csv",
    index=False
)




# ==================================================
# 12. LINEAR REGRESSION, RIDGE AND LASSO
# ==================================================


amount_candidates = [
    "loan_amount",
    "LoanAmount",
    "Loan_Amount",
    "Loan_Amount_Term"
]


regression_target = next(
    (
        col for col in amount_candidates
        if col in df.columns
        and pd.api.types.is_numeric_dtype(df[col])
    ),
    None
)


if regression_target is None:
    print("\nNumeric loan amount column not found.")
    print("Skipping regression models.")
else:


    print("\nRegression target:", regression_target)


    reg_df = df.dropna(subset=[regression_target]).copy()


    X_reg = reg_df.drop(
        columns=[target, regression_target],
        errors="ignore"
    )


    y_reg = reg_df[regression_target]


    reg_numeric = X_reg.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()


    reg_categorical = X_reg.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


    reg_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), reg_numeric),


            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ]), reg_categorical)
        ]
    )


    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=42
    )


    regression_models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=10000)
    }


    regression_results = []


    for name, model in regression_models.items():


        pipeline = Pipeline(steps=[
            ("preprocessor", clone(reg_preprocessor)),
            ("model", model)
        ])


        pipeline.fit(Xr_train, yr_train)


        score = pipeline.score(Xr_test, yr_test)


        coefficients = pipeline.named_steps[
            "model"
        ].coef_


        zero_count = np.sum(
            np.isclose(coefficients, 0)
        )


        regression_results.append({
            "Model": name,
            "R2 Score": score,
            "Zero Coefficients": int(zero_count)
        })


        print("\n", name)
        print("R2 Score:", round(score, 4))
        print("Coefficients:", coefficients)


    regression_results = pd.DataFrame(
        regression_results
    )


    print("\n========== REGRESSION COMPARISON ==========")
    print(regression_results.to_string(index=False))


    regression_results.to_csv(
        OUTPUT_DIR / "regression_results.csv",
        index=False
    )




# ==================================================
# 13. K-MEANS CLUSTERING
# ==================================================


# Select 2 or 3 numeric columns
cluster_columns = (
    df.select_dtypes(include=["int64", "float64"])
    .drop(columns=[target], errors="ignore")
    .var()
    .sort_values(ascending=False)
    .index[:3]
    .tolist()
)


if len(cluster_columns) < 2:
    print("\nNot enough numeric columns for K-Means.")
else:


    print("\nClustering columns:", cluster_columns)


    cluster_data = df[cluster_columns].copy()


    cluster_data = SimpleImputer(
        strategy="median"
    ).fit_transform(cluster_data)


    cluster_data = StandardScaler().fit_transform(
        cluster_data
    )


    # Elbow and Silhouette scores
    inertias = []
    silhouette_scores = []


    max_k = min(8, len(cluster_data) - 1)


    for k in range(2, max_k + 1):


        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )


        labels = kmeans.fit_predict(cluster_data)


        inertias.append(kmeans.inertia_)


        sil_score = silhouette_score(
            cluster_data,
            labels
        )


        silhouette_scores.append(sil_score)


        print(
            f"K={k}, "
            f"Inertia={kmeans.inertia_:.2f}, "
            f"Silhouette={sil_score:.4f}"
        )


    # Elbow curve
    k_values = list(range(2, max_k + 1))


    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow Curve")
    plt.grid(True)


    plt.savefig(
        OUTPUT_DIR / "elbow_curve.png"
    )


    plt.show()


    # Silhouette score plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        k_values,
        silhouette_scores,
        marker="o"
    )


    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("K-Means Silhouette Scores")
    plt.grid(True)


    plt.savefig(
        OUTPUT_DIR / "silhouette_scores.png"
    )


    plt.show()


    # Best K based on highest silhouette score
    best_k = k_values[
        np.argmax(silhouette_scores)
    ]


    final_kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )


    df["Cluster"] = final_kmeans.fit_predict(
        cluster_data
    )


    print("\nBest K:", best_k)


    print("\nCluster descriptions:")
    print(
        df.groupby("Cluster")[cluster_columns]
        .mean()
        .round(2)
    )


    df.to_csv(
        OUTPUT_DIR / "loan_data_with_clusters.csv",
        index=False
    )




# ==================================================
# 14. SAVE MODELS
# ==================================================


import joblib


joblib.dump(
    logistic_pipeline,
    OUTPUT_DIR / "logistic_model.joblib"
)


joblib.dump(
    decision_tree_pipeline,
    OUTPUT_DIR / "decision_tree_model.joblib"
)


joblib.dump(
    random_forest_pipeline,
    OUTPUT_DIR / "random_forest_model.joblib"
)


joblib.dump(
    xgb_pipeline,
    OUTPUT_DIR / "xgboost_model.joblib"
)




# ==================================================
# 15. FINAL OUTPUT
# ==================================================


print("\n======================================")
print("TASK 3 COMPLETED SUCCESSFULLY")
print("======================================")


print("\nFiles saved inside outputs folder:")
print("1. classification_results.csv")
print("2. regression_results.csv (if loan amount exists)")
print("3. elbow_curve.png")
print("4. silhouette_scores.png")
print("5. loan_data_with_clusters.csv")
print("6. Model joblib files")