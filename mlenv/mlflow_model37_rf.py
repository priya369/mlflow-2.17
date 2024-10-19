from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest
from sklearn.model_selection import train_test_split
import mlflow

# Set up MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id="152248560600703153")
mlflow.sklearn.autolog()

# Load California housing dataset
california_housing_data = fetch_california_housing()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    california_housing_data.data, california_housing_data.target, test_size=0.33, random_state=42
)

with mlflow.start_run() as run:
    # Change to Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    model_uri = mlflow.get_artifact_uri("model")

    # Evaluate the model
    result = mlflow.evaluate(
        model_uri,
        X_test,
        targets=y_test,
        model_type="regressor",
        evaluators="default",
        feature_names=california_housing_data.feature_names,
        evaluator_config={"explainability_nsamples": 1000},
    )

# Print metrics and artifacts
print(f"metrics:\n{result.metrics}")
print(f"artifacts:\n{result.artifacts}")
