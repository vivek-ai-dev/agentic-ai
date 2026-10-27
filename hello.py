from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

#client = OpenAI()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1"
)

while True:
    user = input("You: ")
    response = client.responses.create(
        model="openrouter/free",
        input=user
    )
    print(f" Output_Text = {response.output_text}")

"""response = client.responses.create(
    model="openrouter/free",
    input="Explain recursion simply."
)"""

#print(f" OpenAI Version = {openai.__version__}")

#print(f" Output_Text = {response.output_text}")


#print("Hello Agentic AI")