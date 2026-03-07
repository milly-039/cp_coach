import boto3
import os
from dotenv import load_dotenv

load_dotenv()

try:
    sts = boto3.client('sts', region_name='us-east-1')
    identity = sts.get_caller_identity()
    print("✅ AWS Authentication Successful!")
    print(f"Account ID: {identity['Account']}")
    print(f"User ARN: {identity['Arn']}")
except Exception as e:
    print(f"❌ AWS Auth Failed: {e}")