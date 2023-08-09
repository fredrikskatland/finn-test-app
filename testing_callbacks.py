from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain.vectorstores import Pinecone
from langchain.agents import AgentType, initialize_agent

from langchain.tools import tool, Tool


import pickle
import streamlit as st
import pinecone
import os

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])

# initialize pinecone
pinecone.init(
    api_key=st.secrets["PINECONE_API_FINN"], #os.getenv("PINECONE_API_FINN"),  # find at app.pinecone.io
    environment = st.secrets["PINECONE_ENV_FINN"]#os.getenv("PINECONE_ENV_FINN"),  # next to api key in console
)

index_name = "finn-demo-app"
vectorstore = Pinecone.from_existing_index(index_name = index_name, embedding=embeddings)
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever, 
    "search_finn_embeddings",
    "Søk etter relevante stillinger på finn.no."
)

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

st.set_page_config(page_title="Jobbannonser: ", page_icon="🦜")
st.title("🦜 LangChain: Søk på Finn med chat.")


if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("Her kan du søke etter stillinger på finn.no.")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"):
                st.write(step[0].log)
                st.write(f"**{step[1]}**")
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Jeg leter etter lederstillinger innen bank og finans."):
    st.chat_message("user").write(prompt)


    llm = ChatOpenAI(temperature = 0, openai_api_key=st.secrets["openai_api_key"], streaming = True)
    tools =[
        retriever_tool,
    ]
    agent = create_conversational_retrieval_agent(llm, tools, verbose=True)
    #agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors="Check your output and make sure it conforms!", memory=memory)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent(prompt, callbacks=[st_cb])
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]