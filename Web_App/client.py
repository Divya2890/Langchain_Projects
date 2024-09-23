import streamlit as st
import requests

def get_topic_response(input_txt):
    response = requests.post("http://localhost:8000/search_engine/invoke", json={'input':{'topic': input_txt}})
    print("RESPONSE",response)
    return response.json()['output']

st.title("Welcome to Langchain and RAG series")
topic = st.text_input("Ented the topic you want to search for")

if topic:
    st.write(get_topic_response(topic))

