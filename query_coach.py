import json
import os
import boto3
import urllib.request
from pinecone import Pinecone

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("cpcoach")

def get_embedding(text):
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    )
    return json.loads(response['body'].read())['embedding']

def ask_modal(messages):
    url = "https://ishanvi039--ishanvi-cp-coach-cpcoachbrain-generate-hint.modal.run"
    payload = json.dumps({"messages": messages}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode('utf-8'))["answer"]

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        messages = body.get("messages", [])
        
        # Get the latest user question to find the right problem context
        latest_question = messages[-1]["content"]
        
        vector = get_embedding(latest_question)
        search_results = index.query(vector=vector, top_k=2, include_metadata=True)
        
        context_data = "\n".join([m["metadata"]["text"] for m in search_results["matches"]])
        
        # Inject the System Prompt as the FIRST message for the model
        system_msg = {
            "role": "system",
            "content": f"You are a Socratic CP Coach. Source Material:\n{context_data}\n"
                       "If the user is stuck, give a hint. If they have a follow-up doubt, "
                       "explain the logic more deeply using the source material. DO NOT write code."
        }
        
        # Send full history to Modal
        full_payload = [system_msg] + messages
        answer = ask_modal(full_payload)
        
        return {"statusCode": 200, "body": json.dumps({"answer": answer})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}