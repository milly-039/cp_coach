import json
import os
import boto3
#it allows us to talk to bedrock without needing to manage credentials directly, as it will use the AWS credentials configured in the environment
from pinecone import Pinecone
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

def handler(event, context):
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        # UPDATE 1: Your specific CP database
        index = pc.Index("cp-coach")
        
        region = os.getenv("MY_AWS_REGION", "us-east-1")
        bedrock_boto = boto3.client(service_name="bedrock-runtime", region_name=region)

        # Handle both API Gateway/Function URL formats and direct CLI invokes
        body_str = event.get("body", "{}")
        if isinstance(body_str, dict):
            body = body_str
        else:
            body = json.loads(body_str)
            
        # UPDATE 2: Relevant CP fallback question
        question = body.get("question", "I am getting a TLE on my graph traversal, what data structure should I use?")

        # 1. Embed using Titan V2
        emb_res = bedrock_boto.invoke_model(
            modelId="amazon.titan-embed-text-v2:0", 
            body=json.dumps({
                "inputText": question,
                "dimensions": 1024,
                "normalize": True
            })
        )
        query_embedding = json.loads(emb_res['body'].read())['embedding']

        # 2. Query Pinecone
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        context_text = "\n".join([res['metadata']['text'] for res in results['matches']])

        # UPDATE 3: The strict CP Coach System Instructions
        system_prompt = """You are an elite competitive programming coach. Your goal is to help students learn, NOT to do the work for them.
        Read the provided context (problem editorials and hints). 
        When a student asks a question:
        1. Provide conceptual hints, point out logic flaws, or discuss time complexity and appropriate data structures.
        2. UNDER NO CIRCUMSTANCES are you allowed to write or output the final C++, Python, or Java solution code. 
        3. If the context does not contain the answer, rely on standard algorithm principles to guide them."""
        
        user_prompt = f"Context from Editorials:\n{context_text}\n\nStudent Question:\n{question}"
        
        # 3. Generate Answer using Native Nova Lite
        gen_res = bedrock_boto.converse(
            modelId="amazon.nova-lite-v1:0",
            system=[{"text": system_prompt}], # Injecting the strict rules here
            messages=[{
                "role": "user",
                "content": [{"text": user_prompt}]
            }],
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.3 # Kept low so it stays analytical and mathematically precise
            }
        )
        
        answer = gen_res['output']['message']['content'][0]['text']

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }