from openai import OpenAI
from dotenv import load_dotenv

# Load env variables (e.g. OpenAI Key)
load_dotenv()

class ChatGPT:

    def __init__(self):
        self.model = "gpt-4o"

    def call(_self, messages, format=None):

        client = OpenAI()
        if format is None:
            response = client.chat.completions.create(
                model=_self.model,
                temperature=0.2,
                top_p=1,
                presence_penalty=1.1,
                messages=messages
            )
        else:
            response = client.chat.completions.create(
                model=_self.model,
                temperature=0.2,
                top_p=1,
                presence_penalty=1.1,
                messages=messages,
                response_format={ "type": format},
            )
        return response.choices[0].message.content