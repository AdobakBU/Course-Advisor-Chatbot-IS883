import os
import fitz  # PyMuPDF
import csv
from typing import List
import requests
from io import StringIO
from langchain.schema import Document  
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from google.colab import userdata
import streamlit as st

# Load PDF documents using PyMuPDF
folder_path = "https://raw.githubusercontent.com/AdobakBU/Course-Advisor-Chatbot-IS883/main/data/"

def load_pdf_and_csv(folder_path):
    """
    Function to fetch PDF and CSV files from a GitHub folder URL and process them into chunks.

    Args:
        folder_path (str): URL to the folder containing the PDF and CSV files.

    Returns:
        list: A list of Document objects, each representing a chunk of text from the files.
    """
    # List of file names in the folder to process, new file names must be added manually!
    filenames = [
        "2023_FALL_FE_711_A1.pdf", "2023_FALL_FE_711_B1.pdf", "2023_FALL_FE_722_OL.pdf",
        "2023_FALL_FE_822_E1.pdf", "2023_FALL_FE_823_D1.pdf", "2023_FALL_FE_850_D1.pdf",
        "2023_FALL_FE_870_S1.pdf", "2023_FALL_FE_918_A1.pdf", "2023_SPRG_FE_712_A1.pdf",
        "2023_SPRG_FE_712_B1.pdf", "2023_SPRG_FE_712_C1.pdf", "2023_SPRG_FE_713_E1.pdf",
        "2023_SPRG_FE_722_F1.pdf", "2023_SPRG_FE_833_E1.pdf", "2023_SPRG_FE_850_E1.pdf",
        "2023_SPRG_FE_854_E1.pdf", "2023_SPRG_FE_920_A1.pdf"
    ]

    chunks = []

    # Iterate through the list of filenames
    for filename in filenames:
        # This line was not indented causing the error
        file_url = f"{folder_path}{filename}"  
        # Handle PDF files
        if filename.endswith('.pdf'):
            try:
                response = requests.get(file_url)
                response.raise_for_status()  # Raise an error for failed requests

                # Process the PDF
                with fitz.open(stream=response.content, filetype="pdf") as doc:
                    for page_number in range(len(doc)):
                        page = doc.load_page(page_number)
                        text = page.get_text()
                        # Create a Document object for each chunk
                        chunks.append(Document(page_content=text, metadata={"source": filename, "page": page_number + 1}))
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        # Handle CSV files
        elif filename.endswith('.csv'):
            try:
                response = requests.get(file_url)
                response.raise_for_status()  # Raise an error for failed requests

                # Process the CSV
                csv_file = StringIO(response.text)  # Create an in-memory file-like object
                reader = csv.reader(csv_file)
                for row_number, row in enumerate(reader):
                    text = ', '.join(row)  # Combine CSV fields into a single string
                    # Create a Document object for each row
                    chunks.append(Document(page_content=text, metadata={"source": filename, "row": row_number + 1}))
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    # Return the accumulated chunks after the loop finishes
    return chunks


# Retrieve OpenAI API key
openaikey = st.secrets["OpenAI_API_KEY"]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = load_pdf_and_csv(folder_path)

chunks = text_splitter.split_documents(chunks)

# Initialize the FAISS vector store with OpenAI embeddings
openai_api_key = openaikey
faiss_store = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=openai_api_key))

# Define the number of top matching chunks to retrieve
number_of_top_matches = 5

# Prompt the user for a question
question = st.chat_input("What is up?")

# Retrieve top matching chunks from FAISS store
top_matching_chunks = faiss_store.similarity_search_with_score(question, k=number_of_top_matches)

# Combine content from all top matching chunks
combined_context = " ".join([chunk.page_content for chunk, score in top_matching_chunks])

# Answer generation using LangChain's Retrieval-Augmented Generation (RAG) chain
temperature = 1.0
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature)

# Enhanced system prompt for the language model
system_prompt = (
    """
    You are an academic advisor assistant at Boston University. Answer all questions based on the provided context.
    If the answer is not explicitly mentioned in the context, try to summarize the most relevant information related to the question.

    Context:
    {context}

    Question:
    {input}

    Please provide a concise answer based on the above information.
    """
)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the aggregator to assemble documents into a single context
aggregator = create_stuff_documents_chain(llm, prompt=prompt)

# Define the retriever using FAISS store
retriever = faiss_store.as_retriever(k=number_of_top_matches)

# Finalize the RAG chain
rag_chain = create_retrieval_chain(retriever, aggregator)

# Get the answer for the user-provided question
response = rag_chain.invoke({"input": question, "context": combined_context})

# Safely extract the top answer based on the response structure
answer = response.get("answer")

# print the top answer, if not found, indicate it was not available
if answer and 'not explicitly mentioned' not in answer.lower():
    st.write(answer)
else:
    st.write("The specific answer was not found.")
