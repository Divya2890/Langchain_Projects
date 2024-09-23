from fastapi import FastAPI
import uvicorn
from langserve import add_routes
from langchain_core.prompts import PromptTemplate
from langchain import HuggingFacePipeline
from transformers import pipeline
import os
import torch

hf_token = "Input hugging face token to be able to download the model"
lang_token = "Input access token from your langchain account to monitor our application"
## Langmith tracking
os.environ["LANGCHAIN_PROJECT"]="Dummy_Project"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=lang_token
os.environ['HF_TOKEN']= hf_token

app = FastAPI()

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",use_auth_token=hf_token,device_map='auto')

template = PromptTemplate.from_template("What is {topic}")
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map = "auto",
    torch_dtype=torch.float16,
    max_length = 50
)
 
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
chain = template | llm 

@app.get('/index')
def index():
    print("Welcome to Langchain and RAG")

add_routes(
    app,
    chain,
    path="/search_engine",
)

if __name__ == '__main__':
    uvicorn.run(app,host='localhost',port=8000)

