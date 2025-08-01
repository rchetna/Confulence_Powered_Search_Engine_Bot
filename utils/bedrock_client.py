import boto3
import os

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

def get_titan_embedding_model():
    return "amazon.titan-embed-text-v1"

def get_claude_model():
    return "anthropic.claude-3-sonnet-20240229.v1"
