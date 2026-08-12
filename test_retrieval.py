from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

question = "What is the threshold of an artificial neuron?"

results = vector_store.similarity_search(
    question,
    k=3
)

print(f"Question: {question}\n")

for i, document in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print(f"Page: {document.metadata.get('page')}")
    print(document.page_content)
    print()