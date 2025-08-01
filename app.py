import json
import os
import sys
import boto3
import streamlit as st

## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


## Data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="ai21.j2-mid-v1",client=bedrock,
                model_kwargs={'maxTokens':512})
    
    return llm

def get_llama2_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_claude_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_llama2_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()


# import os
# import shutil
# import boto3
# import streamlit as st

# # LangChain & Bedrock
# from langchain_community.embeddings import BedrockEmbeddings
# from langchain.llms.bedrock import Bedrock
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA

# # Bedrock clients
# bedrock = boto3.client(service_name="bedrock-runtime")
# bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# # Function: Ingest PDFs from S3
# def data_ingestion_from_s3(bucket_name, prefix=""):
#     s3 = boto3.client("s3")
#     documents = []
#     os.makedirs("s3_pdfs", exist_ok=True)

#     response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

#     if "Contents" not in response:
#         return []

#     for obj in response["Contents"]:
#         key = obj["Key"]
#         if key.endswith(".pdf"):
#             local_path = os.path.join("s3_pdfs", os.path.basename(key))
#             s3.download_file(bucket_name, key, local_path)

#             loader = PyPDFLoader(local_path)
#             documents.extend(loader.load())

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     docs = text_splitter.split_documents(documents)

#     # Cleanup local files
#     shutil.rmtree("s3_pdfs")

#     return docs

# # Vector Store
# def get_vector_store(docs):
#     vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
#     vectorstore_faiss.save_local("faiss_index")

# # LLM Definitions
# def get_claude_llm():
#     return Bedrock(model_id="anthropic.claude-v2", client=bedrock, model_kwargs={'max_tokens_to_sample': 512})

# def get_llama2_llm():
#     return Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})

# # Prompt template
# # prompt_template = """
# # Human: Using the following context, directly answer the question in a detailed summary of approximately 250 words. 
# # If you don't know the answer, just say that you don't know, don't try to make up an answer.

# # <context>
# # {context}
# # </context>

# # Question: {question}

# # Assistant:"""
# prompt_template = """
# Human: You are a highly factual assistant. Using only the information provided in the <context> tags below, answer the question in a detailed summary of approximately 150 words.

# Do NOT use prior knowledge or assumptions. If the answer is not clearly stated in the context, respond only with: "don't know".

# Avoid any polite phrases or explanations about limitations. Be strictly informative and factual. Do not invent or infer.

# <context>
# {context}
# </context>

# Question: {question}

# Assistant:
# """

# PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# # Run LLM and return response
# def get_response_llm(llm, vectorstore_faiss, query):
#     retriever = vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     docs = retriever.get_relevant_documents(query)

#     if not docs or all(len(doc.page_content.strip()) < 50 for doc in docs):
#         return "I’m sorry, I could not find an answer to your question in the provided documents."

#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT}
#     )
#     answer = qa({"query": query})
#     return answer['result']

# # Streamlit App
# def main():
#     st.set_page_config("Chat PDF from S3", layout="wide")
#     st.header("Chat with PDFs in Amazon S3 using AWS Bedrock 💁‍♀️")

#     user_question = st.text_input("Ask a question from the PDF documents")

#     with st.sidebar:
#         st.title("Update Vector Store from S3:")
#         bucket_name = st.text_input("S3 Bucket Name")
#         prefix = st.text_input("S3 Prefix (folder name or leave blank)", "")

#         if st.button("Update Vectors from S3"):
#             with st.spinner("Fetching and processing PDFs from S3..."):
#                 docs = data_ingestion_from_s3(bucket_name, prefix)
#                 if docs:
#                     get_vector_store(docs)
#                     st.success("Vector store updated from S3 PDFs.")
#                 else:
#                     st.warning("No PDF files found in the specified S3 location.")

#     if st.button("Claude Output"):
#         with st.spinner("Processing with Claude..."):
#             faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
#             llm = get_claude_llm()
#             response = get_response_llm(llm, faiss_index, user_question)
#             st.write(response)
#             st.success("Done")

#     if st.button("Llama2 Output"):
#         with st.spinner("Processing with LLaMA 3..."):
#             faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
#             llm = get_llama2_llm()
#             response = get_response_llm(llm, faiss_index, user_question)
#             st.write(response)
#             st.success("Done")

# if __name__ == "__main__":
#     main()
