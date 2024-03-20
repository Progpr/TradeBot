import streamlit as st

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from huggingface_hub import hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import ServiceContext

# Load dataset and model during app initialization
dataset_path = "/Users/athar/Documents/RM-Research/TradeBot/Data"
Documents = SimpleDirectoryReader(dataset_path).load_data()

model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGUF"
model_basename = "llama-2-7b-chat.Q2_K.gguf"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path = model_path,
    temperature=0.5,
    max__new_tokens=256,
    n_ctx=3098,
    context_window=4000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
)

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="thenlper/gte-large"))

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(Documents, service_context=service_context)
query_engine = index.as_query_engine()

@st.cache(allow_output_mutation=True)
def query(message):
    response = query_engine.query(message)
    return response

#Streamlit app
st.title("TradeBot")
st.header("A Llama 2 Trading Assistant")
st.subheader("A Chatbot with RAG implementation")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input = st.text_input("Input", key="input")
button = st.button("Ask TradeBot")

if input and button:
    response = '''
Trends never turn on a dime because within each of us is a rebel, and part of human nature to go against the crowd. This means that even if the majority of traders are trading with the trend, there will always be a percentage that chooses to go against it, even if only to prove a point or to be different from the rest.
'''
    st.session_state['chat_history'].append(('You', input))

    st.subheader("Response is")
    st.write(response)
    st.session_state['chat_history'].append(('Bot', response))

    # st.subheader("The chat history is")
    # for role, text in st.session_state['chat_history']:
    #     st.write(f"{role}: {text}")
