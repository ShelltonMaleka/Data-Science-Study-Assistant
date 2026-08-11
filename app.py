from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

chat_history = []

print("Data Science Study Assistant")
print("Type 'exit' to quit.")

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    chat_history.append(HumanMessage(content=question))

    response = llm.invoke(chat_history)

    chat_history.append(AIMessage(content=response.content))

    print(f"\nAssistant: {response.content}")