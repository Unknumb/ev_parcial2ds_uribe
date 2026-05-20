import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

logger = logging.getLogger(__name__)

def preprocess_regression(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Para regresión predecimos promedio_notas
    # X (ausencias, semestre, sede, carrera) e y (promedio_notas)
    num_cols = ["total_ausencias", "semestre"]
    cat_cols = ["carrera", "sede"]
    target_col = "promedio_notas"
    
    df_ml = df.dropna(subset=num_cols + cat_cols + [target_col]).copy()
    
    X = df_ml[num_cols + cat_cols]
    y = df_ml[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_regression(X_train: pd.DataFrame, y_train: pd.Series):
    num_cols = ["total_ausencias", "semestre"]
    cat_cols = ["carrera", "sede"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline

def evaluate_regression(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info("=== Regression Evaluation ===")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R2 Score: {r2:.4f}")
