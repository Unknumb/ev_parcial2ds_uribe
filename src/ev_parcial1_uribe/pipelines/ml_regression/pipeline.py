from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_regression, train_regression, evaluate_regression

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_regression,
                inputs="tabla_maestra_primaria",
                outputs=["X_train_reg", "X_test_reg", "y_train_reg", "y_test_reg"],
                name="preprocess_regression_node",
            ),
            node(
                func=train_regression,
                inputs=["X_train_reg", "y_train_reg"],
                outputs="regression_model",
                name="train_regression_node",
            ),
            node(
                func=evaluate_regression,
                inputs=["regression_model", "X_test_reg", "y_test_reg"],
                outputs=None,
                name="evaluate_regression_node",
            ),
        ]
    )
