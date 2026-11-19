from app.chat_model import ChatBot

def main():
    chatbot = ChatBot()

    print("Groq AI Chatbot is ready!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Question : ").strip()

        if user_input.lower() == "exit":
            print("Model Exit")
            break

        if not user_input:
            continue

        try:
            reply = chatbot.get_response(user_input)
            print(f"\nModel : {reply}\n")

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
    