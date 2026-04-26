# src_py/ml/regressor.py
# Phase 5: Target-agnostic Machine Learning Regressor Engine

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class TargetAgnosticRegressor:
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=None, 
            random_state=42, 
            n_jobs=-1
        )
        
    def train_and_evaluate(self, X, Y):
        # 80/20 Train-Test split for model validation
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        
        print("Initiating Random Forest Regression Pipeline...")
        self.model.fit(X_train, Y_train)
        
        # Inference vector generation
        predictions = self.model.predict(X_test)
        
        # Statistical Metric extraction
        mse = mean_squared_error(Y_test, predictions)
        r2 = r2_score(Y_test, predictions)
        
        print(f"ML Pipeline Evaluation -> MSE: {mse:.4e}, R2 Score: {r2:.4f}")
        return self.model, mse, r2