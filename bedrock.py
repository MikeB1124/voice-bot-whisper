import boto3
import json

# Create a session
session = boto3.Session(
    aws_access_key_id="",
    aws_secret_access_key="",
)
bedrock_runtime = session.client("bedrock-runtime", region_name="us-west-2")

# Updated system prompt to request structured JSON
system_prompt = """
Your only job is to translate from one language to another.
If you are given text in English, translate it word for word into Spanish.
If you are given text in Spanish, translate it word for word into English.
Always respond ONLY in this exact JSON format, without any other text:

{
  "translation": "<translated_text>",
  "language": "<language_you_translated_to>"
}

If you cannot translate, return "unknown" as the language and an empty string for translation.
"""

# Example user message (Spanish to English)
user_message = "Hola, mi nombre es Jessica Miguel, quería estaba hablando para saber sobre mi estado de cuenta, ha cambiado mi pago mensual y quiero saber por qué."

# Build the full prompt
prompt_text = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|start_header_id|>user<|end_header_id|>
{user_message}
<|start_header_id|>assistant<|end_header_id|>
"""

# Request body
body_dict = {
    "prompt": prompt_text,
    "max_gen_len": 512,
    "top_p": 0.9,
    "temperature": 0.6,
}

# Invoke model
response = bedrock_runtime.invoke_model(
    modelId="arn:aws:bedrock:us-west-2:934985413136:inference-profile/us.meta.llama3-3-70b-instruct-v1:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(body_dict).encode("utf-8"),
)

# Read and parse the model's response
response_body = json.loads(response["body"].read())

# The model's text output (string) is inside "generation"
generated_text = response_body["generation"]

# Now parse the generated JSON from the model
try:
    result = json.loads(generated_text)
    translation = result.get("translation", "")
    language = result.get("language", "unknown")
    print("Translation:", translation)
    print("Translated To:", language)
except json.JSONDecodeError:
    print("Error: Model response was not valid JSON.")
    print("Raw response:", generated_text)
