"""
Fitness Tracker Data Analysis & Machine Learning
Author: Nurtas Kalmakhan 
Date: 21.03.2025
Description: This script analyzes fitness tracker data, performs EDA, and builds ML models.
"""

# ==================== IMPORTS ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, accuracy_score, 
                            classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

# ==================== DATA LOADING ====================
def load_data():
    """Load and return synthetic fitness data."""
    data = pd.DataFrame([
        [1,'2023-10-01',8523,420,78,'Running',7.5],
        [2,'2023-10-01',6542,320,72,'Walking',6.8],
        [3,'2023-10-01',12045,580,88,'Cycling',7.2],
        [1,'2023-10-02',7432,380,75,'Swimming',6.5],
        [2,'2023-10-02',9876,450,80,'Running',8.0],
        [3,'2023-10-02',5678,290,68,'Walking',7.0],
        [1,'2023-10-03',11234,540,85,'Cycling',7.8],
        [2,'2023-10-03',4321,220,65,'Resting',5.5],
        [3,'2023-10-03',7654,370,74,'Swimming',6.2],
        [1,'2023-10-04',9231,460,82,'Running',7.0],
        [2,'2023-10-04',6789,330,70,'Walking',6.0],
        [3,'2023-10-04',10987,520,87,'Cycling',8.5],
        [1,'2023-10-05',5432,280,67,'Resting',5.8],
        [2,'2023-10-05',8765,420,77,'Swimming',7.3],
        [3,'2023-10-05',12345,590,90,'Running',6.9],
        [1,'2023-10-06',7654,370,73,'Cycling',7.1],
        [2,'2023-10-06',9876,480,83,'Running',8.2],
        [3,'2023-10-06',6543,320,69,'Resting',5.0],
        [1,'2023-10-07',8765,430,79,'Swimming',6.7],
        [2,'2023-10-07',5432,270,66,'Resting',5.5],
        [3,'2023-10-07',11234,550,86,'Cycling',7.4],
        [1,'2023-10-08',9876,490,84,'Running',7.9],
        [2,'2023-10-08',7654,380,76,'Swimming',6.5],
        [3,'2023-10-08',8765,440,81,'Cycling',7.0],
        [1,'2023-10-09',6543,310,70,'Resting',5.2],
        [2,'2023-10-09',10987,530,88,'Running',8.1],
        [3,'2023-10-09',5432,260,64,'Resting',4.8],
        [1,'2023-10-10',11234,560,89,'Cycling',7.6],
        [2,'2023-10-10',7654,390,78,'Swimming',6.9],
        [3,'2023-10-10',9876,470,82,'Running',7.8],
        [1,'2023-10-11',8765,440,80,'Cycling',7.3],
        [2,'2023-10-11',6543,330,71,'Resting',5.7],
        [3,'2023-10-11',5432,250,63,'Resting',4.5],
        [1,'2023-10-12',10987,540,87,'Running',8.0],
        [2,'2023-10-12',7654,370,74,'Swimming',6.4],
        [3,'2023-10-12',8765,430,79,'Cycling',7.2],
        [1,'2023-10-13',9876,480,83,'Running',7.7],
        [2,'2023-10-13',6543,320,68,'Resting',5.3],
        [3,'2023-10-13',11234,570,90,'Cycling',8.3],
        [1,'2023-10-14',7654,380,77,'Swimming',6.8],
        [2,'2023-10-14',8765,450,81,'Cycling',7.5],
        [3,'2023-10-14',5432,260,65,'Resting',4.9],
        [1,'2023-10-15',10987,550,88,'Running',8.4],
        [2,'2023-10-15',9876,490,85,'Running',7.9],
        [3,'2023-10-15',6543,340,72,'Resting',5.6],
        [1,'2023-10-16',11234,580,91,'Cycling',8.6],
        [2,'2023-10-16',7654,400,79,'Swimming',7.0],
        [3,'2023-10-16',8765,460,82,'Cycling',7.4]
    ], columns=["User_ID","Date","Steps","Calories_Burned","Heart_Rate","Activity_Type","Sleep_Hours"])
    
    return data

# ==================== EDA & VISUALIZATION ====================
def perform_eda(data):
    """Perform exploratory data analysis and visualization."""
    print("\n=== Basic Statistics ===")
    print(data.describe())
    
    print("\n=== Activity Distribution ===")
    print(data["Activity_Type"].value_counts())
    
    # Visualization
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Steps vs Calories
    plt.subplot(2, 3, 1)
    sns.scatterplot(x="Steps", y="Calories_Burned", hue="Activity_Type", data=data)
    plt.title("Steps vs Calories Burned")
    
    # Plot 2: Heart Rate Distribution
    plt.subplot(2, 3, 2)
    sns.boxplot(x="Activity_Type", y="Heart_Rate", data=data)
    plt.title("Heart Rate by Activity")
    
    # Plot 3: Sleep Impact
    plt.subplot(2, 3, 3)
    sns.regplot(x="Sleep_Hours", y="Steps", data=data)
    plt.title("Sleep Hours vs Steps")
    
    # Plot 4: Daily Steps Trend
    plt.subplot(2, 3, 4)
    daily_steps = data.groupby("Date")["Steps"].mean()
    daily_steps.plot(kind="line")
    plt.title("Daily Steps Trend")
    plt.xticks(rotation=45)
    
    # Plot 5: Correlation Heatmap
    plt.subplot(2, 3, 5)
    numeric_data = data.select_dtypes(include=[np.number])
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    
    plt.tight_layout()
    plt.savefig("fitness_eda.png")
    plt.show()

# ==================== FEATURE ENGINEERING ====================
def preprocess_data(data):
    """Preprocess data for ML models."""
    # Feature engineering
    data["Date"] = pd.to_datetime(data["Date"])
    data["Day_of_Week"] = data["Date"].dt.day_name()
    
    # Encode categorical variables
    le = LabelEncoder()
    data["Activity_Encoded"] = le.fit_transform(data["Activity_Type"])
    
    # One-hot encode day of week
    data = pd.get_dummies(data, columns=["Day_of_Week"])
    
    return data, le

# ==================== MACHINE LEARNING ====================
def train_ml_models(data, le):
    """Train and evaluate ML models."""
    # Regression: Predict Calories
    X_reg = data[["Steps", "Heart_Rate", "Activity_Encoded", "Sleep_Hours"]]
    y_reg = data["Calories_Burned"]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)
    
    # Classification: Predict Activity
    X_clf = data[["Steps", "Calories_Burned", "Heart_Rate", "Sleep_Hours"]]
    y_clf = data["Activity_Encoded"]
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42)
    
    # Regression Models
    print("\n=== CALORIES BURNED PREDICTION (REGRESSION) ===")
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor()
    }
    
    for name, model in reg_models.items():
        model.fit(X_train_reg, y_train_reg)
        pred = model.predict(X_test_reg)
        rmse = np.sqrt(mean_squared_error(y_test_reg, pred))
        print(f"{name} RMSE: {rmse:.2f}")
    
    # Classification Models
    print("\n=== ACTIVITY TYPE PREDICTION (CLASSIFICATION) ===")
    clf_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    
    for name, model in clf_models.items():
        model.fit(X_train_clf, y_train_clf)
        pred = model.predict(X_test_clf)
        acc = accuracy_score(y_test_clf, pred)
        print(f"\n{name} Accuracy: {acc:.2f}")
        print(classification_report(y_test_clf, pred, target_names=le.classes_))
        
        # Confusion Matrix
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test_clf, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("====== FITNESS TRACKER DATA ANALYSIS ======")
    
    # Load data
    data = load_data()
    print("\nData loaded successfully! Shape:", data.shape)
    
    # EDA
    perform_eda(data)
    
    # Preprocessing
    data, le = preprocess_data(data)
    
    # Machine Learning
    train_ml_models(data, le)
    
    print("\nAnalysis complete! Check generated visualizations.")

