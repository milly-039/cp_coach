import pandas as pd
import json
import ast
import re
import random
import os

def clean_html(raw_html):
    """Removes HTML tags to ensure the AI reads clean text."""
    if pd.isna(raw_html):
        return ""
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, '', str(raw_html))
    return text.replace('&nbsp;', ' ').replace('&lt;', '<').replace('&gt;', '>')

def main():
    print("Loading leetcode.csv...")
    
    # Check if file exists before trying to read
    if not os.path.exists("leetcode.csv"):
        print("❌ Error: leetcode.csv not found in the current folder.")
        return

    df = pd.read_csv("leetcode.csv")

    # Filter for quality: We only want problems that actually have hints to train the AI
    df_train = df.dropna(subset=['hints', 'description', 'title']).copy()

    training_data = []
    success_count = 0

    print("Generating conversational training data...")

    # We will grab 500 high-quality examples to train the model
    for _, row in df_train.head(500).iterrows():
        title = row['title']
        desc = clean_html(row['description'])
        
        # Safely parse the lists stored as strings in the CSV
        try:
            hints = ast.literal_eval(row['hints']) if isinstance(row['hints'], str) else []
            topics = ast.literal_eval(row['topics']) if isinstance(row['topics'], str) else []
        except Exception:
            continue
            
        # Skip if there are no hints
        if not hints or len(hints) == 0:
            continue
            
        # Variation in User Prompts to make the AI robust
        user_prompts = [
            f"I'm stuck on the problem '{title}'. Can you give me a hint?",
            f"How do I approach the '{title}' problem optimally?",
            f"I keep getting TLE on '{title}'. Any hints?",
            f"Help me understand '{title}' without giving me the code."
        ]
        
        user_prompt = random.choice(user_prompts)
        context = f"Problem: {title}\nDifficulty: {row['difficulty']}\nTopics: {', '.join(topics)}\nDescription: {desc}"
        
        # Build the Coach Response (The Socratic Method)
        hint_text = " ".join(hints)
        assistant_response = (
            f"Let's break this down conceptually. The key topics here are {', '.join(topics) if topics else 'core algorithmic patterns'}. "
            f"{hint_text} "
            "Does that spark an idea on how to optimize your time complexity?"
        )
        
        # Format explicitly for Qwen-2.5 ChatML format
        formatted_text = (
            "<|im_start|>system\nYou are a Socratic Competitive Programming Coach. Give conceptual hints. DO NOT write code solutions.<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n\nContext:\n{context}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_response}<|im_end|>"
        )
        
        training_data.append({"text": formatted_text})
        success_count += 1

    # Save to JSONL
    output_file = "train.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Checkpoint cleared! Successfully created {output_file} with {success_count} Socratic training examples.")

if __name__ == "__main__":
    main()