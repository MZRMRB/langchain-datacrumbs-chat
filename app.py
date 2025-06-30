import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load API key safely
class GeminiAPI:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

# Load .env
load_dotenv()
try:
    gemini = GeminiAPI()
    api_key = gemini.api_key
except ValueError as e:
    st.error(f"📛 {e}")
    st.stop()

# Configure Gemini
genai.configure(api_key=api_key)

# Page setup
st.set_page_config(page_title="💬 Chat with Datacrumbs", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 Chat with <span style='color:#6C63FF;'>Datacrumbs</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by Gemini + LangChain</p>", unsafe_allow_html=True)
st.divider()

# Load & split website
@st.cache_resource(show_spinner="🔄 Loading Datacrumbs website...")
def load_website_data():
    loader = WebBaseLoader("https://datacrumbs.org/")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

documents = load_website_data()
st.success(f"✅ Loaded {len(documents)} text chunks from Datacrumbs.org")

# Embeddings & Retriever
embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)
vectorstore = FAISS.from_documents(documents, embedding)
retriever = vectorstore.as_retriever()

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=api_key
)

# LangChain QA setup
chain = load_qa_chain(llm, chain_type="stuff")
qa = RetrievalQA(combine_documents_chain=chain, retriever=retriever, return_source_documents=False)

# Session state for handling example question input
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Input form
with st.form(key="chat_form"):
    question = st.text_input("💭 Ask your question:", value=st.session_state.input_text)
    submit = st.form_submit_button("🔍 Ask")
    if submit:
        st.session_state.input_text = question  # update for re-use

# Suggest realistic questions
st.markdown("💡 **Try one of these questions:**")
suggested_questions = [
    "What kind of bootcamps does Datacrumbs offer?",
    "How can I join the Datacrumbs community?",
    "What skills can I learn at Datacrumbs?"
]
cols = st.columns(len(suggested_questions))
for i, q in enumerate(suggested_questions):
    with cols[i]:
        if st.button(q):
            st.session_state.input_text = q
            st.rerun()

# Determine source
def is_related_to_datacrumbs(q):
    keywords = ["datacrumbs", "bootcamp", "community", "tech", "skills", "platform", "python", "data"]
    return any(word in q.lower() for word in keywords)

# Final Q&A handler
if question:
    with st.spinner("🤖 Thinking..."):
        if is_related_to_datacrumbs(question):
            answer = qa.run(question)
        else:
            response = llm.invoke(question)
            answer = response.content if hasattr(response, "content") else response

    st.toast("✅ Answer generated!")
    st.markdown("### 💬 Answer:")
    st.write(answer)
