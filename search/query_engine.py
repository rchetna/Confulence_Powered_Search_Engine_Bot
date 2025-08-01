# # from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import BedrockEmbeddings
# # from langchain_community.llms import Bedrock
# # from langchain.chains import RetrievalQA

# # from utils.bedrock_client import get_bedrock_client, get_titan_embedding_model, get_claude_model

# # def run_query(question: str):
# #     embeddings = BedrockEmbeddings(
# #         client=get_bedrock_client(),
# #         model_id=get_titan_embedding_model()
# #     )
# #     vectorstore = FAISS.load_local("faiss_index", embeddings)

# #     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# #     llm = Bedrock(
# #         client=get_bedrock_client(),
# #         model_id=get_claude_model(),
# #         model_kwargs={"temperature": 0.5}
# #     )

# #     qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# #     answer = qa.run(question)
# #     return answer

# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import BedrockEmbeddings
# from langchain_community.llms import Bedrock
# from langchain.chains import RetrievalQA

# from utils.bedrock_client import get_bedrock_client, get_titan_embedding_model, get_claude_model

# def run_query(question: str):
#     embeddings = BedrockEmbeddings(
#         client=get_bedrock_client(),
#         model_id=get_titan_embedding_model()
#     )

#     # Allow safe pickle loading
#     vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#     llm = Bedrock(
#         client=get_bedrock_client(),
#         model_id=get_claude_model(),
#         model_kwargs={"temperature": 0.5}
#     )

#     qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

#     answer = qa.run(question)
#     return answer


from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from utils.bedrock_client import get_bedrock_client, get_claude_model, get_titan_embedding_model

def run_query(question: str):
    client = get_bedrock_client()

    embeddings = BedrockEmbeddings(client=client, model_id=get_titan_embedding_model())
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = BedrockChat(
        client=client,
        model_id=get_claude_model(),
        model_kwargs={
            "temperature": 0.5,
            "max_tokens": 1000,
            "anthropic_version": "bedrock-2023-05-31"
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain.run(question)
