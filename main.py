import streamlit as st
from search.query_engine import run_query
from embeddings.embed_and_store import embed_documents
import os

st.set_page_config(page_title="Confluence Semantic Search", page_icon="🔍")

st.title("🔍 Confluence Semantic Search Engine (AWS Bedrock Powered)")

if st.button("🧠 Embed Confluence Pages"):
    with st.spinner("Embedding documents..."):
        embed_documents()
    st.success("Documents embedded and FAISS index created!")

query = st.text_input("Ask a question about your Confluence content:")

if query:
    with st.spinner("Generating answer using Claude..."):
        answer = run_query(query)
    st.markdown("### 🤖 Answer:")
    st.write(answer)
