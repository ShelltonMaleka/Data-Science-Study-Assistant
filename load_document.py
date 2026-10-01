import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader("data/AML_Lecture1.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Number of pages: {len(documents)}")
print(f"Number of chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate stable IDs for each document chunk
chunk_ids = []

for chunk in chunks:

    # Combine document content and metadata
    chunk_identifier = (
        chunk.page_content
        + str(sorted(chunk.metadata.items()))
    )

    # Generate a deterministic hash
    chunk_id = hashlib.sha256(
        chunk_identifier.encode("utf-8")
    ).hexdigest()

    chunk_ids.append(chunk_id)


# Connect to the existing Chroma database
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# Insert or update chunks using their stable IDs
vector_store.add_documents(
    documents=chunks,
    ids=chunk_ids
)

print("Vector database updated successfully!")
print(f"Processed {len(chunks)} document chunks.")
print(f"Total stored chunks: {vector_store._collection.count()}")

print("Vector database created successfully!")