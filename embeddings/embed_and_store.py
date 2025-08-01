from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from ingest.confluence_loader import fetch_confluence_pages
import os
from dotenv import load_dotenv
from utils.bedrock_client import get_bedrock_client, get_titan_embedding_model

load_dotenv()

def embed_documents():
    pages = fetch_confluence_pages()

    docs = [Document(page_content=page["content"], metadata={"title": page["title"]}) for page in pages]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    embeddings = BedrockEmbeddings(
        client=get_bedrock_client(),
        model_id=get_titan_embedding_model()
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ Documents embedded and stored successfully.")
