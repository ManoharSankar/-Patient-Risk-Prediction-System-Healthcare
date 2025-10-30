import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import boto3
import tarfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--s3_bucket', type=str, default='healthcarepatientrisk')
    parser.add_argument('--s3_model_key', type=str, default='models/diabetes-rf')
    args, _ = parser.parse_known_args()

    train_path = 's3://sagemaker-ap-south-1-*/diabetes-readmission/data/train/train.csv'
    test_path = 's3://sagemaker-ap-south-1-*/diabetes-readmission/data/test/test.csv'
    model_dir = os.environ.get('SM_MODEL_DIR', '.')

    print("📥 Loading data...")
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

    print("🚀 Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n✅ Evaluation:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    model_tar_path = os.path.join(model_dir, "model.tar.gz")
    with tarfile.open(model_tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="model.joblib")

    s3 = boto3.client('s3')
    s3.upload_file(model_tar_path, args.s3_bucket, f"{args.s3_model_key}/model.tar.gz")
    print(f"☁️ Uploaded to s3://{args.s3_bucket}/{args.s3_model_key}/model.tar.gz")

    print("✅ Training and upload complete.")
