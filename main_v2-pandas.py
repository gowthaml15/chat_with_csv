
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
from pandasai import PandasAI, OpenAI
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory,ConversationSummaryBufferMemory

import matplotlib
import io

matplotlib.use("TkAgg")



def get_response(llm = None,generated="",question="",df=None,flag=False):
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

# load_dotenv()
# API_KEY = os.environ['OPENAI_API_KEY']
# openai.api_key = API_KEY

st.title("Prompt-driven Analysis by Pandas AI")
open_ai_key = st.text_input("Enter Open AI Key")

if open_ai_key:
    openai.api_key = open_ai_key
    
    llm = OpenAI(model = "gpt-3.5-turbo",temperature=0,openai_api_key=open_ai_key)
    pandas_ai = PandasAI(llm,verbose=True,enable_cache=False)





    multiple_files = st.file_uploader(
        "Multiple File Uploader",
        accept_multiple_files=True
        # type=["csv"]
    )
    count=0
    if multiple_files is not None:
        df = pd.DataFrame()
        dataframe = []
        for file in multiple_files:

            data = io.BytesIO(file.getbuffer())
            st.write(file.name)
            
            dj = pd.read_csv(data)
            dj = preprocessing(dj)
            dataframe.append(dj)
            st.write(dj.head(3))

        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        if 'past' not in st.session_state:
            st.session_state['past'] = []
        
        user_input = get_question()
        if user_input:

            if "plot" in user_input.lower():
                st.write(pandas_ai.run(dj,prompt="As a data analyst bot you have to report to the question, give if the answer if it is in money realted with no more than 2 precision:"+user_input))
                
            else:
                pandas_response = pandas_ai.run(dataframe,prompt=user_input)

                answer = get_response(llm,pandas_response,user_input)
                st.write(pandas_response)
                st.write(answer)


                # st.session_state.past.append(user_input)
                # st.session_state.generated.append(answer)

                # if st.session_state['generated']:

                #     for i in range(len(st.session_state['generated'])-1, -1, -1):
                #         message(st.session_state["generated"][i],is_table=True, key=str(i))
                #         message(st.session_state['past'][i], is_user=True,is_table=True, key=str(i) + '_user')



    

    


    


