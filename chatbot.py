import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA

#loading all the env variable 
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

# Chatprompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful massistant . Please  repsonse to the user queries"),
        ("user","Question:{question}")
    ]
)

#Function to dynamically chats with models 
def get_llm_response(api_key,engine,question):
    os.environ["NVIDIA_API_KEY"] = api_key
    llm = ChatNVIDIA(model=engine)
    output_parser = StrOutputParser()

    chain = prompt|llm|output_parser

    response = chain.invoke({"question":question})

    return response

## App 
st.title("Enhanced Q&A Chatbot With OpenAI")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Open AI API Key:",type="password")

## Select the model
engine=st.sidebar.selectbox("Select model",["nvidia/nemotron-3-ultra-550b-a55b","mistralai/mistral-medium-3.5-128b","moonshotai/kimi-k2.6"])


## MAin interface for user input
st.write("Goe ahead and ask any question")
user_input=st.text_input("You:")

if user_input and api_key:
    response=get_llm_response(api_key,engine,question=user_input)
    st.write(response)

elif user_input:
    st.warning("Please enter the aPi Key in the sider bar")
else:
    st.write("Please provide the user input")





