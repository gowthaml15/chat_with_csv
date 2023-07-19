
import os 
import streamlit as st
import tempfile
import shutil
import openai
import pandas as pd
import pandasai
import datetime

from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from pandasai import PandasAI
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory,ConversationSummaryBufferMemory

import matplotlib
import io

matplotlib.use("TkAgg")

def get_response(generated="",question="",df=None,flag=False):
    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm = llm,
        verbose=True,
        memory=memory
    )
    if flag:
        today_date = datetime.date.today()
        num_rows = len(df)
        num_columns = len(df.columns)
        df_head = df.head()
        prompt = f""""Today is {today_date}.
                    You are provided with a pandas dataframe (df) with {num_rows} rows and {num_columns} columns.
                    This is the metadata of the dataframe:
                    {df_head}.

                    Using the provided dataframe, df, make yourself sumary to answer if questions asked based on data"""
    
    else:
        prompt = f""" 
                As a data analyst bit you have to report the generated answer, give if the answer is in money realted with no more than 2 precision:
                {generated}
                For the question:
                {question}
                """
    response = conversation.predict(input = prompt)
    
    return response

def preprocessing(df):
    # df = pd.read_csv(file_path)
    date_column_list = ["date"]
    for i in df.columns:
        for j in date_column_list:
            if j in i.lower():
                df[['Date', 'Time']] = df['Purchase Date'].str.split(' ', expand=True)
                df[i] = pd.to_datetime(df['Date'],format='%d/%m/%Y')
                df = df.drop('Date',axis=1)
    return df
def get_question():
    return st.text_input("Talk to the bot",key="input_text")

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
openai.api_key = API_KEY

llm = OpenAI(temperature=0)
pandas_ai = PandasAI(llm,verbose=True,enable_cache=False)

st.title("Prompt-driven Analysis by Pandas AI")



multiple_files = st.file_uploader(
    "Multiple File Uploader",
    accept_multiple_files=True,
    type=["csv"]
)
count=0
if multiple_files is not None:
    df = pd.DataFrame()
    dataframe = []
    for file in multiple_files:
        # file_container = st.beta_expander(
        #     f"File name: {file.name} ({file.size})"
        # )
        data = io.BytesIO(file.getbuffer())
        st.write(file.name)
        
        dj = pd.read_csv(data)
        dj = preprocessing(dj)
        # dj = pandas_ai.clean_data(dj)
        dataframe.append(dj)
        st.write(dj.head(3))

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    
    user_input = get_question()
    if user_input:
        # if count==0:
        #     summary_response = get_response_2(df=dataframe[0],flag=True)
            # print("summary response...",summary_response)
            # count+=1

        if "plot" in user_input.lower():
            print("inside plotting")
            st.write(pandas_ai.run(dj,prompt="remove indexes while reporting "+user_input))
            
        else:
            pandas_response = pandas_ai.run(dataframe,prompt="Give me a detailed answer with columns involving with the columns and remove indexes while reporting"+user_input)

            answer = get_response(pandas_response,user_input)

            st.session_state.past.append(user_input)
            st.session_state.generated.append(answer)

            if st.session_state['generated']:

                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state["generated"][i],is_table=True, key=str(i))
                    message(st.session_state['past'][i], is_user=True,is_table=True, key=str(i) + '_user')


    

    


    


