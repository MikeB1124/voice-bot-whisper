import torch
from transformers import pipeline

class Llama3():
    def __init__(self, model_id: str):
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.messages = [
            {"role": "system", "content": "You are NBA basketball history expert, you will be tested and should answer all questions accuratly. Please keep your answers short and to the point, should be couple sentences at most."},
            {"role": "user", "content": ""},
        ]


    def invoke_llama(self, prompt):
        self.messages[1]["content"] = prompt
        outputs = self.pipe(
            self.messages,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
            max_new_tokens=256,
        )
        return outputs[0]["generated_text"][-1]["content"]


