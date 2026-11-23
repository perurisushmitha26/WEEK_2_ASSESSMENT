import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import(
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

#2.Load Dataset
data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
Y=data.target
print("Dataset loaded sucessfully")
print("Dataset Shape:",X.shape)
print("Target Shape:",Y.shape)

#Split data into Train,Validation and test
X_train,X_temp,Y_train,Y_temp=train_test_split(
    X,Y,
    test_size=0.40,
    random_state=42,
    stratify=Y
)
X_val,X_test,Y_val,Y_test=train_test_split(
    X_temp,Y_temp,
    test_size=0.50,
    random_state=42,
    stratify=Y_temp
)
print("\nData Split Results")
print("Training Data:",X_train.shape)
print("Validation data:",X_val.shape)
print("Testing Data:",X_test.shape)
#4.demonstarte overfitting using decision Tree
depths=[1,3,5,10,None]
train_scores=[]
val_scores=[]
print("\nDecision Tree Results")
for depth in depths:
    model=DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train,Y_train)

    #Training accuracy
    train_pred=model.predict(X_train)
    train_acc=accuracy_score(Y_train,train_pred)

    #validation  accuracy
    val_pred=model.predict(X_val)
    val_acc=accuracy_score(Y_val,val_pred)
    train_scores.append(train_acc)
    val_scores.append(val_acc)
    print(
        f"Depth:{depth},"
        f"Train Accuracy:{train_acc:.2f},"
        f"Validation Accusracy:{val_acc:.2f}"
    )
    #5. Plot Train vs validation Accuracy
plt.figure(figsize=(9,5))
plt.plot(
        [str(d) for d in depths],
        train_scores,
        marker='o',
        label='Training Accuracy'
    )
plt.plot(
        [str(d) for d in depths],
        val_scores,
        marker='o',
        label='Validation Accuracy'
    )
plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy -Overfitting")
plt.legend()
plt.grid(True)
plt.show()

    #6.5-Fold Cross Valiadtion usingLogistic Regression
print("\n5-Fold Cross Validation")

logistic_model=make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)
    )
cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
scores=cross_val_score(
        logistic_model,
        X_train,
        Y_train,
        cv=cv,
        scoring='accuracy'
    )
print("Cross Validatin Scores:",scores)
print("Mean Accuracy:",round(scores.mean(),4))
print("Standard Deviation:",round(scores.std(),4))

    #7.Demonstrate Data Leakage
print("\n DATA LEAKAGE DEMONSTRATION")
    #WRONG WAY: Scalig before splitting
scaler_wrong=StandardScaler()
X_scaled_wrong=scaler=scaler_wrong.fit_transform(X)
X_train_wrong,X_temp_wrong,Y_train_wrong,Y_temp_wrong=train_test_split(
        X_scaled_wrong,
        Y,
        test_size=0.40,
        random_state=42,
        stratify=Y
    )
    
X_val_wrong,X_test_wrong,Y_val_wrong,Y_test_wrong=train_test_split(
            X_temp_wrong,
            Y_temp_wrong,
            test_size=0.50,
            random_state=42,
            stratify=Y_temp_wrong
        )
wrong_model=LogisticRegression(max_iter=5000)
wrong_model.fit(X_train_wrong,Y_train_wrong)
wrong_pred=wrong_model.predict(X_val_wrong)
wrong_accuracy=accuracy_score(Y_val_wrong, wrong_pred)
print("Accuracy with Data Leakage:",round(wrong_accuracy,4))
    #CORRECT WAY:SPlit fisrt ,scale using training 
scaler_correct=StandardScaler()
X_train_scaled=scaler_correct.fit_transform(X_train)
X_val_scaled=scaler_correct.transform(X_val)
correct_model=LogisticRegression(max_iter=5000)
correct_model.fit(X_train_scaled,Y_train)
correct_pred=correct_model.predict(X_val_scaled)
correct_accuracy=accuracy_score(Y_val,correct_pred)
print("accuracy without data leakage:",round(correct_accuracy,4))

    #8.Classify Four ML Problem
print("\n MLProblem classification")
problems= {
    "Spam Detection":"Supervised Learning",
    "Customer Segmentation":"Unsupervised Learing",
    "House Price Prediction":"Supervised Learning",
    "Anomaly Detection":"Unsupervised Learning"
}
for problem,learning_type in problems.items():
    print(problem,":",learning_type)

    #9.ML Workflow in 7 steps
print("\n ML Workflow")
workflow=[
    "1.Problem Definition",
    "2.Data Collection",
    "3.Data Preprocessing",
    "4.Train,Validatin and Test Split",
    "5.Model Selection and Training",
    "6.Model Evaluation and Tuning",
    "7.Final Testing and development"
]
for step in workflow:
    print(step)

    #10.final Test Evaluation
final_model=make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)
final_model.fit(X_train,Y_train)
test_predictions=final_model.predict(X_test)
test_accuracy=accuracy_score(Y_test,test_predictions)
print("\nFIAL TEST ACCURACY:",round(test_accuracy,4))
print("\n TASK1 completed")
