"""
Módulo de Entrenamiento de Modelos.
Contiene las funciones para dividir los datos, preprocesarlos y entrenar los modelos (Clasificación y Regresión).
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestRegressor

from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)

def preprocess_and_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocesa y divide los datos para el modelo de clasificación."""
    df['estado_matricula'] = df['estado_matricula'].str.strip().str.upper()
    df['estado_matricula'] = df['estado_matricula'].replace(['CONGELADA', 'DESERTOR'], 'BAJA_RETENCION')
    df['estado_matricula'] = df['estado_matricula'].replace(['REGULAR', 'EGRESADO'], 'NO_RIESGO')
    
    # Target Binario
    target_map = {'NO_RIESGO': 0, 'BAJA_RETENCION': 1}
    df['target_bin'] = df['estado_matricula'].map(target_map)
    
    num_cols = ["total_ausencias", "promedio_notas", "semestre"]
    cat_cols = ["carrera", "sede"]
    target_col = "target_bin"
    
    df_ml = df.dropna(subset=num_cols + cat_cols + [target_col]).copy()
    
    X = df_ml[num_cols + cat_cols]
    y = df_ml[target_col]
    
    # Semilla Óptima congelada (44)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=44, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena el modelo de clasificación LGBMClassifier Binario."""
    num_cols = ["total_ausencias", "promedio_notas", "semestre"]
    cat_cols = ["carrera", "sede"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    pipeline = ImblearnPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('classifier', LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1))
    ])
    
    logger.info("Entrenando Modelo LGBMClassifier Binario...")
    pipeline.fit(X_train, y_train)
    
    return pipeline

def preprocess_regression(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocesa y divide los datos para el modelo de regresión."""
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
    """Entrena el modelo RandomForestRegressor."""
    num_cols = ["total_ausencias", "semestre"]
    cat_cols = ["carrera", "sede"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    pipeline = SklearnPipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline
