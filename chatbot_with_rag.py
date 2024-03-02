import streamlit as st

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

from huggingface_hub import hf_hub_download

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

from llama_index.core import ServiceContext


#dataset
dataset_path = "/Users/athar/Desktop/Research/DATA/"
Documents = SimpleDirectoryReader(dataset_path).load_data()

#model loading
model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGUF"
model_basename = "llama-2-7b-chat.Q2_K.gguf" # the model is in bin format

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)


def query(message):

    # #template
    # template = """Question: {question}

    # Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    # prompt = PromptTemplate.from_template(template)

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.75,
        max__new_tokens=256,
        n_ctx = 3098,
        context_window =4000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    response = llm.invoke(message)

    return response

st.title("TradeBot")
st.header("A Llama 2 Trading Assistant")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input = st.text_input("Input", key="input")
button = st.button("Ask TradeBot")

if input and button:
    response = query(input)
    st.session_state['chat_history'].append(('You', input))

    st.subheader("Response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(('Bot',chunk.text))
    st.subheader("The chat history is")

    for role,text in st.session_state['chat_history']:
        st.write["f{role}:{text}"]


    


