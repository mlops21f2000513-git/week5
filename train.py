REGISTERED_MODEL_NAME="IRIS_classifier_dt_2"
TRAINING_DATA="data/iris.csv"
EXPERIMENT_NAME = "Assgn IRIS Classifier Decision Tree"

# --- ML flow ---
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient(mlflow.get_tracking_uri())
all_experiments = client.search_experiments()

mlflow.set_experiment(EXPERIMENT_NAME)
# --- ML flow ---

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv(TRAINING_DATA)

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

params = {
    "max_depth": 1, "random_state": 15
}

mod_dt = DecisionTreeClassifier(**params)
mod_dt.fit(X_train,y_train)
prediction = mod_dt.predict(X_test)
accuracy = metrics.accuracy_score(prediction, y_test)

# --- ML flow ---
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tag("Training info", "Decision Tree model for IRIS dataset")
    signature = infer_signature(X_train, mod_dt.predict(X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model = mod_dt,
        name = "iris_model",
        signature = signature,
        input_example = X_train,
        registered_model_name = REGISTERED_MODEL_NAME
    )
