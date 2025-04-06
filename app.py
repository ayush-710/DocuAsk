import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from PIL import Image  # For image processing
import pytesseract  # For OCR
import pandas as pd  # For CSV and Excel processing

from pymongo import MongoClient
import tempfile

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load MongoDB connection string from .env
MONGO_DB_CONNECTION_STRING = os.getenv("MONGO_DB_CONNECTION_STRING")


def get_file_text(files):
    text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.name.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(file)
            text += pytesseract.image_to_string(image) + "\n"
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            text += df.to_string(index=False) + "\n"
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
            text += df.to_string(index=False) + "\n"
        else:
            st.warning(f"Unsupported file type: {file.name}")
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, mongo_connection_string, db_name, collection_name):
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the vector store to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store.save_local(temp_dir)

        # Read all files in the temporary directory as binary data
        files_data = {}
        for file_name in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file_name)
            with open(file_path, "rb") as f:
                files_data[file_name] = f.read()

        # Connect to MongoDB
        client = MongoClient(mongo_connection_string)
        db = client[db_name]
        collection = db[collection_name]

        # Clear the collection before inserting the new vector store
        collection.delete_many({})

        # Store the binary data in MongoDB
        collection.insert_one({"vector_store_files": files_data})

        # Close the MongoDB connection
        client.close()

    st.success("Vector store saved to MongoDB successfully!")
    
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, mongo_connection_string, db_name, collection_name):
    # Connect to MongoDB and retrieve the vector store files
    client = MongoClient(mongo_connection_string)
    db = client[db_name]
    collection = db[collection_name]
    vector_store_files = collection.find_one()["vector_store_files"]
    client.close()

    # Save the binary data to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name, file_data in vector_store_files.items():
            file_path = os.path.join(temp_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(file_data)

        # Load the vector store from the temporary directory
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search
    docs = vector_store.similarity_search(user_question)

    # Get the conversational chain
    chain = get_conversational_chain()

    # Generate the response
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])
    
def main():
    st.set_page_config("Chat Files")
    st.header("Chat with Files using Gemini💁")

    # MongoDB database and collection details
    db_name = "faiss_index"
    collection_name = "input_files"

    user_question = st.text_input("Ask a Question from the Files")

    if user_question:
        user_input(user_question, MONGO_DB_CONNECTION_STRING, db_name, collection_name)

    with st.sidebar:
        st.title("Menu:")
        files = st.file_uploader(
            "Upload your Files (PDF, Images, CSV, Excel) and Click on the Submit & Process Button",
            accept_multiple_files=True,
            type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx"]
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_file_text(files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, MONGO_DB_CONNECTION_STRING, db_name, collection_name)
                st.success("Done")


if __name__ == "__main__":
    main()