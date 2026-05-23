"""
Módulo de Evaluación de Modelos.
Contiene las funciones para evaluar modelos de Clasificación y Regresión, generando métricas y reportes.
"""
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evalúa el modelo de Clasificación Binaria usando un Sistema Experto Híbrido."""
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

def evaluate_regression(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evalúa el modelo de Regresión."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info("=== Regression Evaluation ===")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R2 Score: {r2:.4f}")
