REGISTERED_MODEL_NAME="IRIS_classifier_dt"
INFERENCE_DATA="data/iris_inference.csv"
PREDICTIONS="predictions.csv"
EXPERIMENT_NAME = "IRIS Classifier Decision Tree"

# --- ML flow ---
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

print("experiment", experiment)

# Get all runs under this experiment
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],  # Sort by accuracy descending
    max_results=1
)

best_run = runs[0]
print("Best run ID:", best_run.info.run_id)
print("Accuracy:", best_run.data.metrics["accuracy"])

# Load that best model
model_uri = f"runs:/{best_run.info.run_id}/iris_model"
loaded_model = mlflow.sklearn.load_model(model_uri)
print("✅ Model loaded successfully from:", model_uri)
# --- ML flow ---

import pandas as pd
from sklearn import metrics

data = pd.read_csv(INFERENCE_DATA)

X_test = data[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = data.species

prediction = loaded_model.predict(X_test)
prediction_df = pd.DataFrame(prediction, columns=['Prediction'])
prediction_df.to_csv(PREDICTIONS, index=False)

accuracy = metrics.accuracy_score(prediction, y_test)
print("inference accuracy", accuracy)
