from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

print("Data Science Study Assistant")
print("Type 'exit' to quit.")

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = llm.invoke(question)

    print(f"\nAssistant: {response.content}")