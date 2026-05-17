from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_and_split, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_and_split,
                inputs="tabla_maestra_primaria",
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="preprocess_and_split_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="classifier_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classifier_model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
