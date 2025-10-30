import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

ROLE_ARN = "arn:aws:iam::*:role/service-role/AmazonSageMaker-ExecutionRole-20251030T141582"
MODEL_S3_PATH = "s3://healthcarepatientrisk/models/diabetes-rf/model.tar.gz"
ENDPOINT_NAME = "diabetes-risk-endpoint"
REGION = "ap-south-1"


def deploy_model():
    print("🚀 Starting SageMaker deployment...")

    session = sagemaker.Session()

    model = SKLearnModel(
        model_data=MODEL_S3_PATH,
        role=ROLE_ARN,
        entry_point="inference.py",
        framework_version="1.2-1",
        sagemaker_session=session,
        py_version="py3"
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME,
        wait=True
    )

    print(f"✅ Deployment complete. Endpoint: {ENDPOINT_NAME}")
    return predictor


if __name__ == "__main__":
    deploy_model()
