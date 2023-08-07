from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent

st.set_page_config(page_title="Jobbannonser: ", page_icon="🦜")
st.title("🦜 LangChain: Søk med chat.")

import pinecone
from langchain.vectorstores import Pinecone
import os

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])

# initialize pinecone
pinecone.init(
    #api_key=os.getenv("PINECONE_API_FINN"),  # find at app.pinecone.io
    #environment=os.getenv("PINECONE_ENV_FINN"),  # next to api key in console

    api_key=st.secrets["PINECONE_API_FINN"], #os.getenv("PINECONE_API_FINN"),  # find at app.pinecone.io
    environment = st.secrets["PINECONE_ENV_FINN"]#os.getenv("PINECONE_ENV_FINN"),  # next to api key in console
)

index_name = "finn-demo-app"
embedding = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index(index_name = index_name, embedding=embedding)
retriever = vectorstore.as_retriever()

tool = create_retriever_tool(
    retriever, 
    "search_finn_embeddings",
    "Søk etter relevante stillinger på finn.no."
)

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
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

if prompt := st.chat_input(placeholder="Jeg leter etterlederstillinger."):
    st.chat_message("user").write(prompt)


    llm = llm = ChatOpenAI(temperature = 0, openai_api_key=st.secrets["openai_api_key"])
    tools = [tool]
    agent = create_conversational_retrieval_agent(llm, tools, verbose=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent(prompt, callbacks=[st_cb])
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]