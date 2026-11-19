
from app.config import chat_model, client


class ChatBot:
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": ("You are a friendly AI assistant. Answer the questions very clearly and shortly")
            }
        ]

    def get_response(self, user_input):
        self.messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            response = client.chat.completions.create(
                model=chat_model,
                messages=self.messages,
                temperature=0.5,
                max_completion_tokens=1024
            )

            print(f"Response: {response.choices}")
            reply = response.choices[0].message.content
            


            if reply is None:
                reply = "The model returned an empty response."

            self.messages.append({
                "role": "assistant",
                "content": reply
            })

            return reply

        except Exception:
            self.messages.pop()
            raise
