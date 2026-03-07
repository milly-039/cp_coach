import pandas as pd
import json
import os
import re
import time
import boto3
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Clients securely
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("cpcoach")
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    print("✅ Successfully connected to AWS and Pinecone.")
except Exception as e:
    print(f"❌ Connection Error: Check your .env file. Details: {e}")
    exit()

def clean_html(raw_html):
    """Removes HTML tags and handles the 719 blank rows safely."""
    if pd.isna(raw_html):
        return ""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', str(raw_html))

def get_embedding(text):
    """Calls AWS Bedrock to convert text to vectors."""
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    )
    return json.loads(response['body'].read())['embedding']

# 2. Load the Dataset
print("Loading data/leetcode.csv...")
df = pd.read_csv("data/leetcode.csv")
total_rows = len(df)

# Configuration for Rate Limiting and Batching
BATCH_SIZE = 50
current_batch = []
success_count = 0

print(f"Starting ingestion for {total_rows} problems. Do not close this terminal...")

# 3. Process the Data Safely
for i, row in df.iterrows():
    try:
        # Construct the knowledge context
        clean_description = clean_html(row.get('description', ''))
        full_context = f"Problem: {row.get('title', 'Unknown')}\nDifficulty: {row.get('difficulty', 'Unknown')}\nDescription: {clean_description}"
        
        # Rate limit: Pause for 1 second to avoid AWS Throttling (HTTP 429)
        time.sleep(1.0) 
        
        # Get vector from Bedrock
        vector = get_embedding(full_context)
        
        # Prepare the data package for Pinecone
        vector_data = {
            "id": str(row.get('frontendQuestionId', f"custom-{i}")), 
            "values": vector, 
            "metadata": {
                "text": full_context, 
                "title": str(row.get('title', 'Unknown')),
                "difficulty": str(row.get('difficulty', 'Unknown'))
            }
        }
        current_batch.append(vector_data)
        
        # Upload in batches of 50
        if len(current_batch) >= BATCH_SIZE:
            index.upsert(vectors=current_batch)
            success_count += len(current_batch)
            print(f"✅ Upserted batch. Total progress: {success_count}/{total_rows}")
            current_batch = [] # Clear the batch
            
    except Exception as e:
        print(f"⚠️ Failed to process row {i} (Problem: {row.get('title', 'Unknown')}). Error: {e}")
        continue # Skip this row and keep going

# Upload any remaining items in the final partial batch
if current_batch:
    index.upsert(vectors=current_batch)
    success_count += len(current_batch)
    print(f"✅ Upserted final batch. Total progress: {success_count}/{total_rows}")

print(f"\n🎉 INGESTION COMPLETE! Successfully loaded {success_count} problems into 'cpcoach'.")