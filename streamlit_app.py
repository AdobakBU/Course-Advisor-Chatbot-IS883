import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain import hub
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import fitz  # PyMuPDF
import csv
from typing import List
import requests
from io import StringIO

# Show title and description.
st.title("💬 Chatbot")

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
        """Returns an answer based on Boston University syllubi. \
        use this for any questions related to classes or academics."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        chunks = load_pdf_and_csv(folder_path)

        chunks = text_splitter.split_documents(chunks)

        # Initialize the FAISS vector store with OpenAI embeddings
        openai_api_key = openaikey
        faiss_store = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=openai_api_key))

        # Define the number of top matching chunks to retrieve
        number_of_top_matches = 5

        # Prompt the user for a question
        question = input("Please enter your question: ")

        # Retrieve top matching chunks from FAISS store
        top_matching_chunks = faiss_store.similarity_search_with_score(question, k=number_of_top_matches)

        # Combine content from all top matching chunks
        combined_context = " ".join([chunk.page_content for chunk, score in top_matching_chunks])

        # Answer generation using LangChain's Retrieval-Augmented Generation (RAG) chain
        temperature = 1.0
        llm = ChatOpenAI(openai_api_key=st.secrets["OpenAI_API_KEY"], model=model_type)


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
        aggregator = create_stuff_documents_chain(chat, prompt=prompt)

        # Define the retriever using FAISS store
        retriever = faiss_store.as_retriever(k=number_of_top_matches)

        # Finalize the RAG chain
        rag_chain = create_retrieval_chain(retriever, aggregator)

        # Get the answer for the user-provided question
        response = rag_chain.invoke({"input": question, "context": combined_context})

        # Safely extract the top answer based on the response structure
        answer = response.get("answer")

    tools = [datetoday]

    # Now we add the memory object to the agent executor
    # prompt = hub.pull("hwchase17/react-chat")
    # agent = create_react_agent(chat, tools, prompt)
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a helpful assistant."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(chat, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools,  memory=st.session_state.memory, verbose= True)  # ### IMPORTANT to use st.session_state.memory and st.session_state.agent_executor.

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.memory.buffer:
    # if (message.type in ["ai", "human"]):
    st.chat_message(message.type).write(message.content)

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):
    
    # question
    st.chat_message("user").write(prompt)

    # Generate a response using the OpenAI API.
    response = st.session_state.agent_executor.invoke({"input":prompt})['output']

    # response
    st.chat_message("assistant").write(response)
    # st.write(st.session_state.memory.buffer)
