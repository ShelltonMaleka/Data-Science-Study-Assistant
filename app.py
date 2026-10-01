import os

import streamlit as st

from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables from the .env file
load_dotenv()


# Configure the Streamlit page
st.set_page_config(
    page_title="Data Science Study Assistant",
    page_icon="📚",
    layout="wide"
)


# Create the sidebar
with st.sidebar:

    st.title("📚 Study Assistant")

    st.write("### Study Material")

    # Show the document currently available
    st.info("AML_Lecture1.pdf")

    st.write("### Knowledge Sources")

    # Allow the user to enable or disable external web search
    web_search = st.toggle(
        "🌐 Allow Web Search",
        value=False
    )

    if web_search:
        st.success("Web search enabled")
    else:
        st.info("Using uploaded study material only")

    st.divider()

    # Clear the conversation
    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.conversation_history = []

        st.rerun()


# Display the application title
st.title("📚 Data Science Study Assistant")

st.write(
    "Ask questions about your Data Science study material."
)


# Load the RAG components
@st.cache_resource
def load_rag_system():

    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Connect to the existing Chroma database
    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    # Create a retriever for normal document searches
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # Connect to Qwen3 through Ollama
    llm = ChatOllama(
        model="qwen3:1.7b",
        temperature=0
    )

    return vector_store, retriever, llm


# Load the RAG system
vector_store, retriever, llm = load_rag_system()


# Create the Tavily client
def create_tavily_client():

    # Get the API key from the environment
    api_key = os.getenv("TAVILY_API_KEY")

    # Return nothing if the key does not exist
    if not api_key:
        return None

    return TavilyClient(
        api_key=api_key
    )


# Store displayed chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []


# Store conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        # Display sources when available
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📖 Sources"):

                for source in message["sources"]:

                    # Display uploaded PDF sources
                    if source["type"] == "pdf":

                        st.markdown(
                            f"📄 **{source['name']}**  \n"
                            f"Page {source['page']}"
                        )

                    # Display web sources
                    elif source["type"] == "web":

                        st.markdown(
                            f"🌐 **[{source['title']}]"
                            f"({source['url']})**"
                        )


# Create the chat input
question = st.chat_input(
    "Ask a question..."
)


# Process the question
if question:

    # Display the user's question
    with st.chat_message("user"):

        st.markdown(question)


    # Save the user's question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Search the uploaded study material
    search_results = vector_store.similarity_search_with_score(
        question,
        k=3
    )
        # DEBUG: Inspect document retrieval
    # st.write("### Retrieval Debugging")

    # st.write("Documents in Chroma:", vector_store._collection.count())

    # for i, (document, score) in enumerate(search_results, start=1):
    #     st.write(f"Document {i}")
    #     st.write(f"Distance score: {score:.4f}")
    #     st.write(f"Content: {document.page_content[:300]}")
    #     st.divider()

    # Set the starting similarity threshold
    similarity_threshold = 1.4


    # Keep documents that are sufficiently similar
    relevant_documents = [
        document
        for document, score in search_results
        if score <= similarity_threshold
    ]


    # Store the PDF context
    document_context_parts = []


    # Store structured sources
    sources = []


    # Build the PDF context
    for document in relevant_documents:

        # Get the page number
        page = document.metadata.get(
            "page",
            "Unknown"
        )

        # Convert zero-based page numbers to normal PDF pages
        if isinstance(page, int):
            page += 1


        # Add the document to the context
        document_context_parts.append(
            f"[Source: AML_Lecture1.pdf, Page {page}]\n"
            f"{document.page_content}"
        )


        # Add the PDF source
        pdf_source = {
            "type": "pdf",
            "name": "AML_Lecture1.pdf",
            "page": page
        }


        # Avoid duplicate PDF sources
        if pdf_source not in sources:

            sources.append(
                pdf_source
            )


    # Combine the relevant PDF chunks
    document_context = "\n\n".join(
        document_context_parts
    )


    # Determine whether the PDF contains relevant information
    pdf_is_relevant = len(
        relevant_documents
    ) > 0


    # Start with an empty web context
    web_context = ""


    # Search the web only when the PDF is not relevant
    if not pdf_is_relevant and web_search:

        tavily_client = create_tavily_client()


        if tavily_client is None:

            st.error(
                "TAVILY_API_KEY was not found. "
                "Check your .env file."
            )


        else:

            with st.spinner(
                "The study material does not contain "
                "enough information. Searching the web..."
            ):

                try:

                    # Search Tavily for external information
                    search_response = (
                        tavily_client.search(
                            query=question,
                            search_depth="basic",
                            max_results=5
                        )
                    )


                    # Get the web search results
                    web_results = (
                        search_response.get(
                            "results",
                            []
                        )
                    )


                    # Store web context
                    web_context_parts = []


                    for result in web_results:

                        # Get the result title
                        title = result.get(
                            "title",
                            "Unknown"
                        )

                        # Get the result content
                        content = result.get(
                            "content",
                            ""
                        )

                        # Get the result URL
                        url = result.get(
                            "url",
                            ""
                        )


                        # Add the result to the context
                        web_context_parts.append(
                            f"[Web Source]\n"
                            f"Title: {title}\n"
                            f"Content: {content}\n"
                            f"URL: {url}"
                        )


                        # Create a structured web source
                        web_source = {
                            "type": "web",
                            "title": title,
                            "url": url
                        }


                        # Avoid duplicate web sources
                        if web_source not in sources:

                            sources.append(
                                web_source
                            )


                    # Combine web results
                    web_context = "\n\n".join(
                        web_context_parts
                    )


                except Exception as error:

                    st.error(
                        f"Web search failed: {error}"
                    )


    # Determine which source Qwen3 can use
    if pdf_is_relevant:

        context = (
            "UPLOADED STUDY MATERIAL:\n\n"
            + document_context
        )

        source_instruction = (
            "Use the uploaded study material "
            "to answer the question."
        )


    elif web_context:

        context = (
            "EXTERNAL WEB SOURCES:\n\n"
            + web_context
        )

        source_instruction = (
            "The uploaded study material did not "
            "contain enough relevant information. "
            "Use the external web sources to answer "
            "the question."
        )


    else:

        context = ""

        source_instruction = (
            "There is not enough information in the "
            "available sources to answer the question."
        )


    # Build the conversation history
    history = "\n".join(
        f"User: {user_message}\n"
        f"Assistant: {assistant_message}"
        for user_message, assistant_message
        in st.session_state.conversation_history
    )


    # Create the final prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a Data Science Study Assistant.

{source_instruction}

Answer the question using ONLY the available sources.

Do not use your general knowledge as evidence.

Do not invent information.

If the available sources do not contain the answer,
say that you cannot find the answer in the available
sources.

Conversation history:
{history}

Available sources:
{context}

Question:
{question}

Answer:"""
    )


    # Format the prompt
    messages = prompt.format_messages(
        source_instruction=source_instruction,
        history=history,
        context=context,
        question=question
    )


    # Generate the answer using Qwen3
    with st.spinner("Thinking..."):

        response = llm.invoke(
            messages
        )


    # Get the answer text
    answer = response.content


    # Save the conversation
    st.session_state.conversation_history.append(
        (
            question,
            answer
        )
    )


    # Save the assistant message and sources
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )


    # Display the answer
    with st.chat_message("assistant"):

        st.markdown(answer)


        # Display sources
        if sources:

            with st.expander("📖 Sources"):

                for source in sources:

                    # Display PDF sources
                    if source["type"] == "pdf":

                        st.markdown(
                            f"📄 **{source['name']}**  \n"
                            f"Page {source['page']}"
                        )

                    # Display web sources
                    elif source["type"] == "web":

                        st.markdown(
                            f"🌐 **[{source['title']}]"
                            f"({source['url']})**"
                        )