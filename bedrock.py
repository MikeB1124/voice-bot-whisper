import boto3
import json

session = boto3.Session("", "")
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')


system_prompt = "Your only job is to translate from one language to another. If you are given text in English you should translate it word for word to Spanish. If you are given text in Spanish you should translate it word for word to English. DO NOT add any other information to your response or answer any questoins."

user_message = "Hola, mi nombre es Jessica Miguel, quería estaba hablando para saber sobre mi estado de cuenta, ha cambiado mi pago mensual y quiero saber por qué."

prompt_text = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|start_header_id|>user<|end_header_id|>
{user_message}
<|start_header_id|>assistant<|end_header_id|>
"""

body_dict = {
    "prompt": prompt_text,
    "max_gen_len": 512,
    "top_p": 0.9,
    "temperature": 0.6
}

response = bedrock_runtime.invoke_model(
    modelId="arn:aws:bedrock:us-west-2:934985413136:inference-profile/us.meta.llama3-3-70b-instruct-v1:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(body_dict).encode("utf-8")
)

response_body = json.loads(response["body"].read())
print(response_body)

