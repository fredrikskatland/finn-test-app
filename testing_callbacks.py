from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
import pickle
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent

from bs4 import BeautifulSoup
import requests

def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)

import xmltodict

r = requests.get("https://www.finn.no/feed/job/atom.xml?rows=200")
xml = r.text
raw = xmltodict.parse(xml)

pages = []
for info in raw['feed']['entry']:
    url = info['link']['@href']
    if 'https://www.finn.no/' in url:
        pages.append({'text': extract_text_from(url), 'source': url})

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
docs, metadatas = [], []
for page in pages:
    splits = text_splitter.split_text(page['text'])
    docs.extend(splits)
    metadatas.extend([{"source": page['source']}] * len(splits))
    print(f"Split {page['source']} into {len(splits)} chunks")

import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)



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


    llm = llm = ChatOpenAI(temperature = 0, openai_api_key=st.secrets["openai_api_key"])
    tools = [tool]
    agent = create_conversational_retrieval_agent(llm, tools, verbose=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent(prompt, callbacks=[st_cb])
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]