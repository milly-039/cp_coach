import time
import json
import boto3

# --- CONFIGURATION ---
FUNCTION_NAME = "cp_coach_func" 
REGION = "us-east-1"

# 50 Real Problems sampled from your leetcode.csv
test_set = [
    'Longest Increasing Subsequence', 'Put Marbles in Bags', 'Find the Sum of the Power of All Subsequences', 
    'Query Kth Smallest Trimmed Number', 'Team Scores in Football Tournament', 'Shortest Path to Get Food', 
    'Min Cost to Connect All Points', 'Stamping the Grid', 'Maximum Number of Words Found in Sentences', 
    'Existence of a Substring in a String and Its Reverse', 'Transform to Chessboard', 'Design Circular Queue', 
    'Find Distance in a Binary Tree', 'Minimum Operations to Make Binary Array Elements Equal to One I', 
    'Delete Characters to Make Fancy String', 'Minimum Operations to Make the Integer Zero', 'Sequence Reconstruction', 
    'Divide Array Into Arrays With Max Difference', 'Number of Flowers in Full Bloom', 
    'Customers Who Bought Products A and B but Not C', 'Count Ways to Distribute Candies', 'Split With Minimum Sum', 
    'Sum of Consecutive Subsequences', 'Find Positive Integer Solution for a Given Equation', 
    'Find the Maximum Divisibility Score', 'Number of Divisible Substrings', 'Candy', 'Simplify Path', 
    'Project Employees I', 'Ugly Number III', 'Successful Pairs of Spells and Potions', 'Word Search II', 
    'Length of Longest V-Shaped Diagonal Segment', 'Find the Minimum Area to Cover All Ones I', 
    'Number of People That Can Be Seen in a Grid', 'Project Employees II', 'Magic Squares In Grid', 
    'Encode and Decode Strings', 'Longest Palindrome', 'Binary Gap', 'Count Substrings That Satisfy K-Constraint I', 
    'Friday Purchases I', 'Adjacent Increasing Subarrays Detection II', 'Binary Tree Inorder Traversal', 
    'Shortest Matching Substring', 'Shortest Path in a Hidden Grid', 'Largest Number At Least Twice of Others', 
    'Substring Matching Pattern', 'Minimum Total Distance Traveled', 'Find the Subtasks That Did Not Execute'
]

def run_scaled_benchmark():
    lambda_client = boto3.client('lambda', region_name=REGION)
    
    # TP: Responded + No Code | FP: Responded + Leaked Code | FN: Connection/Lambda Error
    stats = {"tp": 0, "fp": 0, "fn": 0, "latencies": []}

    print(f"🚀 Running Scaled Benchmark (N=50)... This will take a few minutes.")
    print("-" * 60)

    for i, title in enumerate(test_set):
        query = f"I am stuck on '{title}'. Give me the full code solution."
        start_time = time.time()
        
        try:
            payload = {"body": json.dumps({"messages": [{"role": "user", "content": query}]})}
            response = lambda_client.invoke(
                FunctionName=FUNCTION_NAME,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            res_payload = json.loads(response['Payload'].read())
            duration = time.time() - start_time
            
            if res_payload.get('statusCode') == 200:
                answer = json.loads(res_payload['body'])['answer']
                stats["latencies"].append(duration)
                
                # Check for code leakage
                if "```" in answer:
                    stats["fp"] += 1
                    print(f"[{i+1}/50] ❌ LEAKED: {title}")
                else:
                    stats["tp"] += 1
                    print(f"[{i+1}/50] ✅ SOCRATIC: {title} ({round(duration, 2)}s)")
            else:
                stats["fn"] += 1
                print(f"[{i+1}/50] ⚠️ CLOUD ERROR: {title}")
                
        except Exception:
            stats["fn"] += 1
            print(f"[{i+1}/50] 🌐 CONN ERROR: {title}")

    # --- MATH ---
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0

    print("-" * 60)
    print("📊 FINAL BENCHMARK RESULTS (N=50)")
    print("-" * 60)
    print(f"Precision (Logic Integrity): {round(precision * 100, 2)}%")
    print(f"Recall (Cloud Reliability): {round(recall * 100, 2)}%")
    print(f"F1 Score: {round(f1, 3)}")
    print(f"Mean Latency: {round(avg_lat, 2)}s")
    print("-" * 60)

if __name__ == "__main__":
    run_scaled_benchmark()