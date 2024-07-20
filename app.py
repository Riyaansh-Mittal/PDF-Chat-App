import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

#load the api key
load_dotenv()
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

#function for reading pdf files
def read_pdf(pdf):
    text = ""
    for file in pdf:
        pdf_read = PdfReader(file)
        for page in pdf_read.pages:
            text += page.extract_text()
    return text

#Document chunking
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 80)
    chunks = text_splitter.split_text(text)
    return chunks


#Create Embeddings Store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

#Create Conversation Chain
def get_conversation_chain_pdf():
    prompt_template = """
    Your role is to be a meticulous researcher. Answer the question using only the information found within the context.
    Be detailed, but avoid unnecessary rambling.
    if you cannot find the answer, simply state 'answer is not available in the context'.
    Context: \n{context}?\n
    Question: \n{question}?\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature = 0.3)
    prompt = PromptTemplate(template = prompt_template, imput_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

#Processing User Input
def user_input(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    load_vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = load_vector_db.similarity_search(user_query)
    chain = get_conversation_chain_pdf()

    response = chain(
        {"input_documents":docs, "question": user_query},
        return_only_outputs = True
    )
    print(response)
    st.write("AI_Response", response["output_text"])

def main():
    #st.set_page_config("PDF Chat App")
    st.header("Chat with your PDF files using Google Gemini Pro")
    user_query = st.text_input("Ast question for the PDF File?")
    if user_query:
        user_input(user_query)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF File, and click Submit!", accept_multiple_files=True)
        if st.button("Submit!"):
            with st.spinner('Processing...'):
                raw_text = read_pdf(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Done!")

if __name__ == "__main__":
    main()