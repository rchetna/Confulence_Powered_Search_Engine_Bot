
# 🧠 Confluence Semantic Search Engine with AWS Bedrock

This project builds a **semantic search engine powered by Confluence content**, AWS Bedrock LLMs (Claude/Titan), LangChain, and FAISS.

## 🚀 Features

- 🔗 Connects to Confluence using REST API
- 🧱 Embeds documents using Amazon Titan Embeddings (via AWS Bedrock)
- 🧠 Answers queries using Claude 3 Sonnet (via AWS Bedrock)
- 🔍 Uses FAISS for fast local vector similarity search

---

## 🗂 Folder Structure

```
confluence-search-app/
│
├── ingest/
│   └── confluence_loader.py          # Loads content from Confluence
│
├── embeddings/
│   └── embed_and_store.py            # Embeds & stores chunks in FAISS
│
├── search/
│   └── query_engine.py               # Search & answer with Claude
│
├── utils/
│   └── bedrock_client.py             # AWS Bedrock clients (Titan/Claude)
│
├── .env                              # Your secrets (Confluence + AWS)
├── requirements.txt                  # Python dependencies
└── main.py                           # Entry point
```

---

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your credentials:

```
CONFLUENCE_BASE_URL=https://your-domain.atlassian.net/wiki
CONFLUENCE_EMAIL=your-email@example.com
CONFLUENCE_API_TOKEN=your-confluence-api-token
AWS_REGION=us-east-1
```

### 3. Run the App

```bash
streamlit run main.py
```

---

## 💬 Example

```
❓ Ask your question (or type 'exit'): What is the onboarding process?
🤖 Answer: The onboarding process includes steps such as account creation, team assignment, etc.
```

---

## 🧱 Built With

- [LangChain](https://github.com/langchain-ai/langchain)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Confluence API](https://developer.atlassian.com/cloud/confluence/rest/)

---

## 📌 Future Improvements

- Add OpenSearch or Pinecone for scalable vector store
- Add FastAPI or Streamlit UI for user interface
- Schedule Confluence sync via cron or Airflow

---

## 📄 License

This project is open source under the MIT license.
