import joblib
import mlflow
import mlflow.data
import pandas as pd
import numpy as np
import os
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Configure MinIO (hardcode)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://localhost:9000"
os.environ["MINIO_ACCESS_KEY_ID"] = "z39xoj1EQy8GYWxHfWwq"
os.environ["MINIO_SECRET_ACCESS_KEY"] = "2Y8NKBRxLxAWtYBqL7xOqjRK8YUwCgf8ltq8S1Wh"

# Set mlflow tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment('Lazada Reviews Classifications')

# Load data with smaller sample size
x_train_vec = joblib.load('../data/processed/x_train_vec_1.pkl')
y_train = joblib.load('../data/processed/y_train_1.pkl')

# Take a smaller subset of data
SAMPLE_SIZE = 0.2
x_train_small, _, y_train_small, _ = train_test_split(
    x_train_vec, y_train, 
    test_size=1-SAMPLE_SIZE, 
    random_state=42
)

# Free up memory
del x_train_vec
del y_train

# Create and train model with smaller dataset
logreg = LogisticRegression(max_iter=100)
logreg.fit(x_train_small, y_train_small)

# Predict
y_pred = logreg.predict(x_train_small)

# Get metrics
metrics = classification_report(y_train_small, y_pred, output_dict=True)
accuracy = metrics["accuracy"]

# Create a smaller DataFrame for logging
sample_data = pd.DataFrame({
    'predictions': y_pred,
    'actual': y_train_small
}).head(100)  # Only log 100 rows

# MLflow logging
with mlflow.start_run(run_name="Memory Optimized Run"):
    # Log smaller dataset
    dataset = mlflow.data.from_pandas(
        sample_data,
        source="s3://mlops-lazada/20191002-reviews.csv",
        targets="actual",
        name="lazada_reviews_small",
        predictions="predictions"
    )
    
    # Log parameters and metrics
    mlflow.log_params({
        'sample_size': SAMPLE_SIZE,
        'max_iter': logreg.max_iter,
        'solver': logreg.solver
    })
    
    mlflow.log_metric("accuracy", accuracy)
    
    # Log input with smaller dataset
    mlflow.log_input(dataset, "training")
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=logreg,
        artifact_path="model",
        registered_model_name="Optimized_LogReg_Small",
    )
    
    # Add tags
    mlflow.set_tags({
        "dataset_size": "small",
        "experiment_type": "memory_optimized"
    })