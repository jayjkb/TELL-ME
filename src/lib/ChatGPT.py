from dotenv import load_dotenv
from openai import OpenAI


class ChatGPT:

    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "gpt-4o"

    def call(_self, messages, format=None):

        client = OpenAI(api_key=_self.api_key)
        if format is None:
            response = client.chat.completions.create(
                model=_self.model,
                temperature=0.2,
                top_p=1,
                presence_penalty=1.1,
                messages=messages,
            )
        else:
            response = client.chat.completions.create(
                model=_self.model,
                temperature=0.2,
                top_p=1,
                presence_penalty=1.1,
                messages=messages,
                response_format={"type": format},
            )
        return response.choices[0].message.content
