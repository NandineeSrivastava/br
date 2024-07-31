

import sys
import json
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import certifi
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

client = MongoClient(os.getenv("mongo_uri"), tlsCAFile=certifi.where())
dbName = "Fin-ai"
collections_name = "fin_books"
collection = client[dbName][collections_name]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorStore = MongoDBAtlasVectorSearch(
    collection, embeddings, index_name="default", embedding_key="embedding"
)

def get_conversational_chain():
    prompt_template = """
    You are an expert in stock market education. Answer the following question in the context of stock market education:

    context:\n{context}\n
    question:\n{question}\n
    
    answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.15)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm, retriever=vectorStore.as_retriever(), chain_type_kwargs={"prompt": prompt}
    )
    return chain

def query_data(user_question, concept):
    try:
        full_question = f"Regarding the concept of '{concept}': {user_question}"
        docs = vectorStore.similarity_search(full_question, k=10)
        context = docs[0].page_content if docs else ""
        chain = get_conversational_chain()
        response = chain({"context": context, "query": full_question}, return_only_outputs=True)
        return response["result"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read())
    user_question = input_data.get('userQuestion')
    concept = input_data.get('concept')
    result = query_data(user_question, concept)
    print(json.dumps({"result": result}))
