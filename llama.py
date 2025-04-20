import torch
from transformers import pipeline
import boto3
import json


class LLama3():
    def __init__ (self, model_id: str):
        self.modelId = model_id
        session = boto3.Session("", "")
        self.bedrock_runtime = session.client('bedrock-runtime', region_name='us-west-2')
        self.system_prompt = "Your only job is to translate from one language to another. If you are given text in English you should translate it word for word to Spanish. If you are given text in Spanish you should translate it word for word to English. DO NOT add any other information to your response or answer any questoins."

    def build_prompt(self, user_message: str, system_prompt: str):
        prompt_text = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|start_header_id|>user<|end_header_id|>
        {user_message}
        <|start_header_id|>assistant<|end_header_id|>
        """
        return prompt_text
    
    def invoke_bedrock(self, user_message: str):
        prompt_text = self.build_prompt(user_message, self.system_prompt)
        body_dict = {
            "prompt": prompt_text,
            "max_gen_len": 512,
            "top_p": 0.9,
            "temperature": 0.6
        }

        response = self.bedrock_runtime.invoke_model(
            modelId=self.modelId,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body_dict).encode("utf-8")
        )

        response_body = json.loads(response["body"].read())
        return response_body["generation"].strip()

