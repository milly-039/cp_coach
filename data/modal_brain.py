import modal
from pydantic import BaseModel
from typing import List

app = modal.App("ishanvi-cp-coach")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "transformers", "accelerate", "peft", "fastapi[standard]"
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.cls(image=image, gpu="T4", scaledown_window=120)
class CPCoachBrain:
    
    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base_model, "Ishanvi108/cpcoach-qwen-v1")
        print("✅ Brain is Awake with Memory.")

    @modal.web_endpoint(method="POST")
    def generate_hint(self, request: ChatRequest):
        import torch
        
        # Convert Pydantic objects to a list of dicts for the tokenizer
        chat_history = [{"role": m.role, "content": m.content} for m in request.messages]

        # Use the official chat template (This prevents 'made up' formatting)
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=450,      # Increased for more detailed explanations
                temperature=0.6,         # Balanced: creative but grounded
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the new assistant response
        answer = full_text.split("assistant")[-1].strip()
            
        return {"answer": answer}