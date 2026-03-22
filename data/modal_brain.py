import modal
from pydantic import BaseModel

# 1. Define the Modal App
app = modal.App("ishanvi-cp-coach")

# 2. Setup the Cloud Environment (Install necessary AI libraries)
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "transformers", "accelerate", "peft", "fastapi[standard]"
)

# 3. Define the Models
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_MODEL = "Ishanvi108/cpcoach-qwen-v1"

class PromptRequest(BaseModel):
    prompt: str

# 4. Create the Serverless GPU Class
@app.cls(image=image, gpu="T4", scaledown_window=120)
class CPCoachBrain:
    
    @modal.enter()
    def load_model(self):
        """This runs once when the container wakes up to load the model into GPU memory."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        
        print("📥 Loading Base Qwen Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
        )
        
        print(f"🧠 Applying Ishanvi's Custom Logic ({ADAPTER_MODEL})...")
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
        print("✅ Brain is Awake and Ready.")

    @modal.web_endpoint(method="POST")
    def generate_hint(self, request: PromptRequest):
        """This is the actual API endpoint that AWS Lambda will call."""
        import torch
        
        # We format the incoming text into Qwen's native ChatML structure
        formatted_prompt = (
            "<|im_start|>system\nYou are a Socratic Competitive Programming Coach. "
            "Give conceptual hints. DO NOT write code solutions.<|im_end|>\n"
            f"<|im_start|>user\n{request.prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,      # Enough for a good conceptual hint
                temperature=0.3,         # Keep it focused and logical, not overly creative
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean the output
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's reply
        if "assistant\n" in full_text:
            answer = full_text.split("assistant\n")[-1].strip()
        else:
            answer = full_text.strip()
            
        return {"answer": answer}