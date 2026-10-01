# 📚 Data Science Study Assistant

### A Local AI-Powered Study Companion Using Hybrid Retrieval-Augmented Generation (RAG)

An AI-powered study assistant designed to help students interact with academic material through natural language.

Built using **Python, Streamlit, LangChain, ChromaDB, Hugging Face, Ollama and Tavily**, the application combines document retrieval with a locally running Large Language Model (LLM) to generate context-aware answers supported by identifiable sources.

Unlike a conventional chatbot, the assistant prioritises uploaded study material and optionally retrieves external information when relevant content cannot be found.

---

## 🎯 Project Overview

Students often spend considerable time searching through lengthy lecture notes to find specific explanations or concepts.

The Data Science Study Assistant addresses this problem by implementing a Retrieval-Augmented Generation (RAG) pipeline that retrieves relevant document passages before generating an answer.

The project explores how semantic search, vector databases, local language models and external information retrieval can be integrated into a practical educational application.

### Key Features

- **Document-based question answering:** Retrieve relevant information from indexed lecture notes.
- **Local LLM inference:** Generate answers using Qwen3:1.7b through Ollama.
- **Semantic search:** Convert document chunks into vector embeddings using Hugging Face.
- **Hybrid RAG:** Optionally search the internet through Tavily when relevant document context is unavailable.
- **Source attribution:** Display PDF page references and external website links.
- **Conversation history:** Maintain contextual information across interactions within a session.
- **Duplicate prevention:** Generate deterministic SHA-256 chunk identifiers to prevent repeated ingestion of identical document chunks.
- **Interactive interface:** Communicate with the assistant through Streamlit.

---

## 🏗️ System Architecture

```text
                Student Question
                       |
                       v
             Streamlit Interface
                       |
                       v
              Embedding Model
             all-MiniLM-L6-v2
                       |
                       v
                ChromaDB
             Similarity Search
                       |
              Relevant context?
                 /          \
               YES           NO
                |             |
                v             v
          PDF Context    Web Search Enabled?
                |           /       \
                |         YES        NO
                |          |          |
                |          v          |
                |     Tavily API      |
                |          |          |
                v          v          v
              Context Selection
                       |
                       v
                 Qwen3:1.7b
                 via Ollama
                       |
                       v
              Generated Response
                       |
                       v
              Answer + Sources
```

The application follows a retrieval-first approach. The language model is instructed to generate answers using the available retrieved sources rather than relying on unsupported general knowledge.

---

## 🛠️ Technology Stack

| Technology       | Purpose                                     |
| ---------------- | ------------------------------------------- |
| Python           | Core application development                |
| Streamlit        | Interactive chatbot interface               |
| LangChain        | LLM and retrieval integration               |
| Ollama           | Local language model inference              |
| Qwen3:1.7b       | Response generation                         |
| Hugging Face     | Sentence embeddings                         |
| all-MiniLM-L6-v2 | Semantic text representation                |
| ChromaDB         | Vector storage and similarity search        |
| Tavily API       | External information retrieval              |
| SHA-256          | Deterministic document chunk identification |
| python-dotenv    | Environment variable management             |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ShelltonMaleka/Data-Science-Study-Assistant.git

cd Data-Science-Study-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the project's required Python packages:

```bash
pip install streamlit langchain-core langchain-ollama langchain-chroma langchain-huggingface sentence-transformers pypdf tavily-python python-dotenv
```

A version-pinned `requirements.txt` will be added as part of the project's reproducibility improvements.

### 4. Install Ollama

Download Ollama from:

[https://ollama.com/download](https://ollama.com/download)

Pull the language model:

```bash
ollama pull qwen3:1.7b
```

Verify that the model is available:

```bash
ollama list
```

### 5. Configure Tavily (Optional)

Create a `.env` file in the project directory:

```env
TAVILY_API_KEY=your_api_key_here
```

Obtain an API key from [https://www.tavily.com/](https://www.tavily.com/).

**Security:** Never commit your `.env` file or actual API credentials to GitHub.

The assistant can operate using indexed study material without Tavily. External search requires a valid API key.

### 6. Prepare the vector database

Place the supported lecture PDF in the project's `data/` directory and update the ingestion script's document path if necessary.

Run:

```bash
python load_document.py
```

This processes the lecture material, generates embeddings and stores the resulting chunks in ChromaDB.

### 7. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

[http://localhost:8501](http://localhost:8501)

---

## 🧪 Functional Testing

The application has been tested using academic and general-knowledge questions.

| Test Scenario                                       | Observed Behaviour                                                           |
| --------------------------------------------------- | ---------------------------------------------------------------------------- |
| Perceptron training question                        | Retrieved relevant lecture passages and generated an explanation             |
| General-knowledge question with web search disabled | Reported insufficient information in the available sources                   |
| General-knowledge question with web search enabled  | Retrieved external information through Tavily and generated a sourced answer |
| Repeated PDF ingestion                              | Stored chunk count remained at 17 after duplicate-prevention changes         |

### Duplicate Ingestion Improvement

During development, repeated ingestion of the same lecture PDF resulted in 34 stored chunks instead of the original 17.

The ingestion pipeline was modified to use deterministic SHA-256 chunk identifiers.

After rebuilding the database, the stored chunk count returned to 17, and retrieval results no longer repeated the previously duplicated passages.

These results represent initial functional testing, not a comprehensive evaluation of retrieval accuracy or hallucination rates.

---

## 📈 Future Improvements

- [ ] Implement dynamic PDF uploading.
- [ ] Support multiple lecture documents.
- [ ] Introduce document management and deletion.
- [ ] Evaluate and calibrate similarity thresholds.
- [ ] Implement retrieval-quality evaluation.
- [ ] Measure response latency and CPU utilisation.
- [ ] Add a Study Mode for automatically generated practice questions.
- [ ] Improve conversation memory.
- [ ] Add automated tests.
- [ ] Create a version-pinned requirements file.
- [ ] Explore deployment options for local and hosted inference.

---

## 🔒 Privacy and Limitations

The project uses a locally running LLM through Ollama, reducing reliance on external hosted inference services.

However, enabling Tavily sends search queries to an external service.

Current limitations include:

- Document indexing is configured around a predefined lecture PDF.
- Retrieval relevance depends on a manually configured similarity threshold.
- Local inference performance depends on available hardware.
- The assistant may generate incorrect or incomplete answers; users should verify important academic information against the original sources.

---

## 👨‍💻 Author

**Shellton Maleka**

Postgraduate Diploma in Data Science
University of the Witwatersrand, Johannesburg, South Africa

GitHub: [@ShelltonMaleka](https://github.com/ShelltonMaleka)

*Developed as a practical exploration of Natural Language Processing, Retrieval-Augmented Generation, semantic search and local AI systems.*


