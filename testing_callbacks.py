from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st
import pickle
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent

st.set_page_config(page_title="Jobbannonser: ", page_icon="🦜")
st.title("🦜 LangChain: Søk med chat.")


def configure_retriever(vector_store_path):
    with open(vector_store_path, "rb") as f:
        store = pickle.load(f)
    retriever = store.as_retriever()

    return retriever

retriever = configure_retriever("faiss_store.pkl")

tool = create_retriever_tool(
    retriever, 
    "search_finn",
    "Søk etter relevante stillinger på finn.no."
)
tools = [tool]

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


    llm = llm = ChatOpenAI(temperature = 0)
    tools = [tool]
    agent = create_conversational_retrieval_agent(llm, tools, verbose=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent(prompt, callbacks=[st_cb])
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]