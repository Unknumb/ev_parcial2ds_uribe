import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# IMPORTANTE: Usar el pipeline de imblearn para que SMOTE se aplique correctamente en CV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

def preprocess_and_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Unificación de clases
    df['estado_matricula'] = df['estado_matricula'].replace(['CONGELADA', 'DESERTOR'], 'BAJA_RETENCION')
    
    num_cols = ["total_ausencias", "promedio_notas", "semestre"]
    cat_cols = ["carrera", "sede"]
    target_col = "estado_matricula"
    
    df_ml = df.dropna(subset=num_cols + cat_cols + [target_col]).copy()
    
    X = df_ml[num_cols + cat_cols]
    y = df_ml[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    num_cols = ["total_ausencias", "promedio_notas", "semestre"]
    cat_cols = ["carrera", "sede"]
    
    # Paso 1: ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # Paso 2 y 3 en Pipeline de imblearn
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(max_depth=10, min_samples_split=5, random_state=42, class_weight='balanced'))
    ])
    
    param_grid = {
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Implementación de GridSearchCV con scoring='f1_macro'
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"=== Mejores Parámetros GridSearchCV ===")
    logger.info(grid_search.best_params_)
    
    # Retornamos el pipeline ganador
    return grid_search.best_estimator_

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info("=== Classification Report ===")
    logger.info("\n" + report)
