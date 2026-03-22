import json
import os
import boto3
import time
import urllib.request
from pinecone import Pinecone

# Initialize Cloud Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("cpcoach")

def get_embedding(text):
    """Still uses AWS Titan to convert the user's question into vector numbers for Pinecone."""
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    )
    return json.loads(response['body'].read())['embedding']

def ask_ai(prompt_text):
    """Sends the Pinecone context to Ishanvi's Custom Brain on Modal instead of Nova."""
    url = "https://ishanvi039--ishanvi-cp-coach-cpcoachbrain-generate-hint.modal.run"
    
    # Package the prompt into JSON
    payload = json.dumps({"prompt": prompt_text}).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    
    # Create the HTTP Request
    req = urllib.request.Request(url, data=payload, headers=headers)
    
    try:
        # We set a 60-second timeout because Serverless GPUs sometimes take 5-10 seconds to "wake up" (cold start)
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("answer", "Error: Brain returned an empty response.")
    except Exception as e:
        return f"Coach Brain Connection Error: {str(e)}"

def lambda_handler(event, context):
    try:
        start_time = time.time() # Start MLOps Latency Timer
        
        # Parse Request
        body = json.loads(event.get("body", "{}"))
        question = body.get("question", "How do I optimize a nested for-loop?")

        # RAG Step 1: Vector Search
        vector = get_embedding(question)
        search_results = index.query(vector=vector, top_k=3, include_metadata=True)

        # RAG Step 2: Extract Context
        context_text = ""
        problem_titles = []
        for match in search_results.get("matches", []):
            context_text += match["metadata"].get("text", "") + "\n\n"
            problem_titles.append(match["metadata"].get("title", "Unknown"))

        # RAG Step 3: Generate Response using the Custom LLM
        final_prompt = f"Context:\n{context_text}\n\nStudent: {question}\n\nProvide a conceptual hint."
        answer = ask_ai(final_prompt)
        
        end_time = time.time() # Stop Timer

        # MLOps Observability: Structured JSON Logging
        telemetry_log = {
            "log_type": "RAG_Telemetry",
            "question": question,
            "retrieved_context": problem_titles,
            "latency_seconds": round(end_time - start_time, 2)
        }
        print(json.dumps(telemetry_log)) 

        return {
            "statusCode": 200,
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        print(json.dumps({"log_type": "System_Crash", "error": str(e)}))
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}