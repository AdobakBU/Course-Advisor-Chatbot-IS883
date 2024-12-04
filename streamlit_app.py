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

# Show title and description.
st.title("💬 Chatbot")

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
            "2023_SPRG_FE_854_E1.pdf", "2023_SPRG_FE_920_A1.pdf", "merged_class_schedule.csv",
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
    faiss_store = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=openaikey))

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

    tools = [datetoday, rag_tool]
    
    # Now we add the memory object to the agent executor
    # prompt = hub.pull("hwchase17/react-chat")
    # agent = create_react_agent(chat, tools, prompt)
    from langchain_core.prompts import ChatPromptTemplate
    st.session_state.prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
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
if user_input := st.chat_input("What is up?"):
    
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

    # Finalize the RAG chain
    #rag_chain = create_retrieval_chain(retriever, aggregator)

    # Get the answer for the user-provided question
    #response = rag_chain.invoke({"input": question, "context": combined_context})

    # Safely extract the top answer based on the response structure
    #answer = response.get("answer")            #????? how to reconcile this with answer output below ??? 

    # question
    st.chat_message("user").write(user_input)

    # Generate a response using the OpenAI API.
    ## ??????? do we still need this given response line above ???????????
    response = st.session_state.agent_executor.invoke({"input": user_input}) ## "context": combined_context

    # response
    st.chat_message("assistant").write(response)
    # st.write(st.session_state.memory.buffer)