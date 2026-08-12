from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to the Chroma vector database
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# Create the retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# Connect to Qwen3 through Ollama
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# Create the RAG prompt
prompt = ChatPromptTemplate.from_template(
    """You are a Data Science Study Assistant.

Answer the user's question using ONLY the provided study material.

Use the conversation history to understand follow-up questions.
Do not use information from the conversation history as evidence
unless it is also supported by the study material.

If the answer cannot be found in the study material,
say that you cannot find the answer in the provided material.

Conversation history:
{history}

Study material:
{context}

Question:
{question}

Answer:"""
)


# Store the conversation history
conversation_history = []


print("Data Science Study Assistant")
print("Ask questions about your study material.")
print("Type 'exit' to quit.\n")


while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Assistant: Goodbye! Good luck with your studies.")
        break

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Build the study context
    context_parts = []

    for document in documents:
        page = document.metadata.get("page", "Unknown")
        page_number = page + 1 if isinstance(page, int) else page

        context_parts.append(
            f"[Source: AML_Lecture1.pdf, Page {page_number}]\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Format the conversation history
    history = "\n".join(
        f"User: {user_message}\nAssistant: {assistant_message}"
        for user_message, assistant_message in conversation_history
    )

    # Create the complete prompt
    messages = prompt.format_messages(
        history=history,
        context=context,
        question=question
    )

    # Generate the answer
    response = llm.invoke(messages)

    answer = response.content

    # Save the conversation
    conversation_history.append(
        (question, answer)
    )

    # Display the answer
    print(f"\nAssistant: {answer}")

    # Display the sources used for retrieval
    sources = []

    for document in documents:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")

        if isinstance(page, int):
            page += 1

        source_info = f"{source} - Page {page}"

        if source_info not in sources:
            sources.append(source_info)

    print("\nSources:")

    for source in sources:
        print(f"- {source}")

    print()