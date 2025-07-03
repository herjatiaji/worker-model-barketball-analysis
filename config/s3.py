import boto3
from boto3.session import Session
from mypy_boto3_s3.client import S3Client
from botocore.config import Config
from config.settings import settings

session: Session = boto3.session.Session()
s3_client: S3Client = session.client(
    's3',
    endpoint_url=settings.S3_ENDPOINT,
    aws_access_key_id=settings.S3_ACCESS_KEY_ID,
    aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
    region_name=settings.S3_REGION,
    config=Config(signature_version='s3v4'),
    
)
