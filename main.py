import openai
import re
import httpx
import os
import streamlit as st
import pickle as pkl
from joblib import load
import sklearn

from dotenv import load_dotenv

from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")

# initialise the OpenAI client
client = OpenAI()

# Create a class -Agent- to handle the LLMs
class Agent:
    def __init__(self, system=''):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role":'system', 'content':system})
            
            
    def __call__(self, message):
        self.messages.append({'role':'user', 'content':message})
        result = self.execute()
        self.messages.append({'role':'assistant', 'content':result})
        return result
    
    def execute(self):
        chat_completion = client.chat.completions.create(
            model='gpt-4o',
            temperature=0,
            messages = self.messages
        )
        
        return chat_completion.choices[0].message.content
    
# Open the documents
with open('System prompt.txt') as file:
    system_prompt = file.read()

with open('Health Recommendation Doc.txt') as file:
    health_prompt = file.read()
    

# Function to Process the text coming from the LLM to extract the JSON
def extract_users_details(response):
    import json as js
    import re
    pattern = r"\{(.*?)\}"
    
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        json_str = match.group()
        json_obj = js.loads(json_str)
    return json_obj

# Function to call the built model and makes prediction with it
def predict(llm_json):
    import pickle as pkl
    import numpy as np

    
    job_model = load('Job_model.joblib')
        
    # get the users data
    users_details = extract_users_details(llm_json)
    input_data = np.array([list(users_details.values())])

    # Do prediction
    prediction = job_model.predict(input_data)
    
    # return prediction
    return prediction


# Function to get the user's data
def get_data(users_complain):
    interact = Agent(system_prompt)
    get_complain = interact(users_complain)
    return get_complain

def recommendation(prediction):
    prescribe = Agent(health_prompt)
    input_ = f"The prediction made by the model from the patient's data is {prediction}, can you give a recommendation based on this?"
    give_prescription = prescribe(input_)
    st.write(give_prescription)



# ------------Streamlite Code -------------------
st.title('Maleria Test and Recommendation System')

users_complain = st.text_area('Please, give your complain here')

run = st.button('Check')
if run:
    compalin = get_data(users_complain)
    # prediction chain
    st.write(compalin)
    recommend = recommendation(predict(compalin))
