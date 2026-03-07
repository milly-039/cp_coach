import json
import os
import boto3
from pinecone import Pinecone

# Initialize AWS Bedrock and Pinecone
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("cpcoach") # IMPORTANT: Points to the new dataset

def get_embedding(text):
    """Converts the user's question into math."""
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    )
    return json.loads(response['body'].read())['embedding']

def ask_ai(prompt_text):
    """Sends the context and question to the Amazon Nova Lite model."""
    response = bedrock.invoke_model(
        modelId="amazon.nova-lite-v1:0",
        body=json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt_text}]}],
            "system": [{"text": "You are a world-class Competitive Programming Coach. You help students understand LeetCode problems."}],
            "inferenceConfig": {"max_new_tokens": 1000, "temperature": 0.5}
        })
    )
    return json.loads(response['body'].read())['output']['message']['content'][0]['text']

def lambda_handler(event, context):
    """The main engine that AWS Lambda triggers."""
    try:
        # 1. Parse the incoming web request
        body = json.loads(event.get("body", "{}"))
        question = body.get("question", "How do I optimize a nested for-loop?")

        # 2. Convert question to vector and search Pinecone
        vector = get_embedding(question)
        search_results = index.query(
            vector=vector, 
            top_k=3, 
            include_metadata=True
        )

        # 3. Extract the LeetCode context
        context_text = ""
        for match in search_results.get("matches", []):
            context_text += match["metadata"].get("text", "") + "\n\n"

        # 4. The Strict Coach Prompt
        final_prompt = (
            f"Here is the context of the LeetCode problem(s) the student is asking about:\n"
            f"<context>\n{context_text}\n</context>\n\n"
            f"Student's Question: {question}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Do not give the exact code solution right away. Act as a coach.\n"
            f"2. Give a conceptual hint.\n"
            f"3. Mention the optimal Time and Space complexity they should aim for.\n"
            f"4. Ask them if they want the full solution after they try."
        )

        # 5. Generate Answer
        answer = ask_ai(final_prompt)

        return {
            "statusCode": 200,
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }