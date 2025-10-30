import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sagemaker
import boto3
import io


def preprocess_and_upload(
    s3_bucket='healthcarepatientrisk',
    s3_key='diabetic_data.csv',
    prefix='diabetes-readmission/data'
):
    session = sagemaker.Session()
    s3 = boto3.client('s3')

    print(f"📥 Loading dataset from s3://{s3_bucket}/{s3_key}")
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    print(f"✅ Loaded dataset: {df.shape}")

    df = df[df['gender'] != 'Unknown/Invalid']
    df = df[df['discharge_disposition_id'] != 11]
    df = df.replace('?', np.nan)
    df = df.drop(['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr'], axis=1)
    df = df.drop_duplicates()

    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    df['age'] = df['age'].map({
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }).fillna(0).astype(int)

    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    print(f"✅ After preprocessing: {df.shape}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv('train.csv', index=False, header=False)
    test_df.to_csv('test.csv', index=False, header=False)

    bucket = session.default_bucket()
    train_s3 = session.upload_data('train.csv', bucket=bucket, key_prefix=f'{prefix}/train')
    test_s3 = session.upload_data('test.csv', bucket=bucket, key_prefix=f'{prefix}/test')

    print("✅ Uploaded preprocessed data to S3:")
    print("Train:", train_s3)
    print("Test :", test_s3)

    return train_s3, test_s3


if __name__ == "__main__":
    preprocess_and_upload()
