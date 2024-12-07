import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain import hub
import os
import requests
from PyPDF2 import PdfReader
from langchain.schema import Document
import csv
from typing import List
import requests
from io import StringIO
from io import BytesIO
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import Document  
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

LOGO_URL_LARGE = "https://raw.githubusercontent.com/AdobakBU/Course-Advisor-Chatbot-IS883/main/data/bu-logo.png"

#Show logo 
st.logo(LOGO_URL_LARGE, size='large')

#show sidebar
with st.sidebar:
    st.write("Here are some example prompts to get you started:")
    st.write("-------------------------------------------------")
    st.write("+ 'Please suggest a BU Finance course to me that \
             meets on Mondays and Wednesdays and is at a \
             graduate intro level.' ")
    st.write("+ 'Are there any graduate finance courses offered \
             that have an open book exam policy?' ")
    st.write("+ 'Can you please compare FE722 and FE712 and \
             explain why I might take one course over the other?' ")


# Show title and description.
st.title("ClassQuest-GPT")
st.subheader("For help with BU class discovery and comparison")

# Define the number of top matching chunks to retrieve
number_of_top_matches = 5

# Retrieve OpenAI API key
openaikey = st.secrets["OpenAI_API_KEY"]

### Important part.

# Create a session state variable to flag whether the app has been initialized.
# This code will only be run first time the app is loaded.
if "memory" not in st.session_state: ### IMPORTANT.
    model_type="gpt-4o-mini"

    # initialize the momory
    max_number_of_exchanges = 10
    st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=max_number_of_exchanges, return_messages=True) ### IMPORTANT to use st.session_state.memory.
    
    # LLM
    chat = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)

    # tools
    from langchain.agents import tool
    from datetime import date
    @tool
    def datetoday(dummy: str) -> str:
        """Returns today's date, use this for any \
        questions that need today's date to be answered. \
        This tool returns a string with today's date.""" #This is the desciption the agent uses to determine whether to use the time tool.
        return "Today is " + str(date.today())
    
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
            "2023_SPRG_FE_854_E1.pdf", "2023_SPRG_FE_920_A1.pdf", "merged_class_schedule_with_descriptions.csv",
        ]

        chunks = []

        # Iterate through the list of filenames
        for filename in filenames:
            file_url = f"{folder_path}{filename}"
            
            # Handle PDF files
            if filename.endswith('.pdf'):
                try:
                    response = requests.get(file_url)
                    response.raise_for_status()  # Raise an error for failed requests
                    
                    # Use PyPDF2 to read the PDF content
                    reader = PdfReader(BytesIO(response.content))
                    for page_number, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:  # Ensure there's text to avoid empty chunks
                            chunks.append(Document(
                                page_content=text,
                                metadata={"source": filename, "page": page_number + 1}
                            ))
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
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = load_pdf_and_csv(folder_path)

    chunks = text_splitter.split_documents(chunks)

    # Initialize the FAISS vector store with OpenAI embeddings
    with st.spinner('Constructing BU Course Context...'):
        faiss_store = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=openaikey))

    with st.spinner('Generating FAISS store... Almost done...'):
        if "faiss_store" not in st.session_state:
            st.session_state.faiss_store = FAISS.from_documents(
                chunks, OpenAIEmbeddings(openai_api_key=openaikey)
            )

    retriever = st.session_state.faiss_store.as_retriever(k=number_of_top_matches)
    st.session_state.retriever = retriever

    rag_tool = create_retriever_tool(
    retriever,
    "CourseFileRAG",
    "Searches course description files",
    )

    tools = [datetoday, rag_tool]
    
    # Now we add the memory object to the agent executor
    # prompt = hub.pull("hwchase17/react-chat")
    # agent = create_react_agent(chat, tools, prompt)
    from langchain_core.prompts import ChatPromptTemplate
    st.session_state.prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an academic advisor assistant \
             at Boston University. Answer all questions based \
             on the provided context. If the answer is not \
             explicitly mentioned in the context, try to \
             summarize the most relevant information \
             related to the question. \
             Ensure all answers reflect the following: \
             1. Use a friendly, conversational tone. \
             2. Don't use technical terms such as context for the RAG database in your response. Refer to the context as the BU course library. \
             3. Speak in the first person about looking at the BU course library for answers as a helpful academic advisor. Don't say the BU course library provides or doesn't provide an answer. \
             4. Don't answer questions not related to school. Suggest the user find another resource and to let you know if they have any questions about school. \
             5. Don't hallucinate. If you don't know a factual detail, say so. \
             Context: {context}"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(chat, tools, st.session_state.prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools,  memory=st.session_state.memory, verbose= True)  # ### IMPORTANT to use st.session_state.memory and st.session_state.agent_executor.

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.memory.buffer:
    # if (message.type in ["ai", "human"]):
    st.chat_message(message.type).write(message.content)


# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if user_input := st.chat_input("What BU course questions can I help with?"):
    
    # Prompt the user for a question
    #question = prompt ##???????????????? how to reconcile this? with user input below ?????????????????? 

    # Retrieve top matching chunks from FAISS store
    top_matching_chunks = st.session_state.faiss_store.similarity_search_with_score(user_input, k=number_of_top_matches)

    # Combine content from all top matching chunks
    combined_context = " ".join([chunk.page_content for chunk, score in top_matching_chunks])

    # Answer generation using LangChain's Retrieval-Augmented Generation (RAG) chain
    temperature = 1.0
    llm = ChatOpenAI(openai_api_key=openaikey, temperature=temperature)

    # Create the aggregator to assemble documents into a single context
    aggregator = create_stuff_documents_chain(llm, prompt=st.session_state.prompt)

    # question
    st.chat_message("user").write(user_input)

    st.session_state.memory.input_key = "input"
    st.session_state.memory.output_key = "output"

    # Generate a response using the OpenAI API.
    ## ??????? do we still need this given response line above ???????????
    with st.spinner('Thinking...'):
        response = st.session_state.agent_executor.invoke({"input": user_input, "context": combined_context})

    # response
    st.chat_message("assistant").write(response['output'])
    # st.write(st.session_state.memory.buffer)
