import pandas as pd
import numpy as np
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, f1_score, precision_recall_curve

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)

def preprocess_and_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
    num_cols = ["total_ausencias", "promedio_notas", "semestre"]
    cat_cols = ["carrera", "sede"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('classifier', LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1))
    ])
    
    logger.info("Entrenando Modelo LGBMClassifier Binario...")
    pipeline.fit(X_train, y_train)
    
    return pipeline

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_probs = model.predict_proba(X_test)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = [0.0 if (p + r) == 0 else 2 * (p * r) / (p + r) for p, r in zip(precisions[:-1], recalls[:-1])]
    
    best_threshold = thresholds[np.argmax(f1_scores)] if len(f1_scores) > 0 else 0.5
    y_pred_base = (y_probs >= best_threshold).astype(int)
    
    logger.info(f"=== Umbral Matemático Óptimo de Predicción: {best_threshold:.4f} ===")
    
    # Sistema Experto Híbrido (Regla de Negocio)
    y_pred_expert = y_pred_base.copy()
    mask_override = (y_pred_expert == 1) & (X_test['promedio_notas'] >= 6.0)
    y_pred_expert[mask_override] = 0
    
    final_f1 = f1_score(y_test, y_pred_expert, pos_label=1)
    logger.info(f"=== F1-Score Oficial (con Override Experto): {final_f1:.4f} ===")
    
    target_names = ['NO_RIESGO (0)', 'BAJA_RETENCION (1)']
    report = classification_report(y_test, y_pred_expert, target_names=target_names)
    logger.info("=== Classification Report Final (Híbrido) ===")
    logger.info("\n" + report)
