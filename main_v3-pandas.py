
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


def question_analyiser(question):
    llm = OpenAI(temperature=0.0)
    prompt = f"""Your task is to classify which type of question is that from the mentioned class,The class are 'sentence' and 'table' and 'plot'. The question will be given by the user. {question} answer in one word like 'table' or 'sentence' or  'plot"""
    llmchain = ConversationChain(llm=llm)
    response = llmchain.predict(input=prompt)
    return response

def question_rephraser(dataframe,question,format):
    
    llm = OpenAI(temperature=0.0)
    if "table" in format:
        print("table prompt...")
        prompt = f""" Your task is to rephrase the question meaningful for the chat to answer for the below question tag and giving the columns related to that question. Additionally rephrasing the question the this {format} format. Additionaly, rephrase if ID involved, provide the description or name from the co-responding thing for that id.
        <question> {question} </question>
                """

    elif "sentence" in format:
        print("sentence prompt...")
        prompt = f""" Your task is to rephrase the question in the question tag 
         <question> {question} </question> 
         give the response in double quotes"""


    llmchain = ConversationChain(llm=llm)
    response = llmchain.predict(input=prompt)
    pandas_ai = PandasAI(llm=llm,verbose=True,enable_cache=False)
    response = pandas_ai.run(dataframe,"Give me a detailed answer with columns involving with the columns and remove indexes while reporting"+response)

    return "response ---> "+response


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
    st.write(user_input)
    if user_input:
        format = question_analyiser(question=user_input)

        # if "table" in format.lower():
        #     st.write(question_rephraser(dataframe,user_input,"table"))
        # elif "sentence" in format.lower():
        #     st.write(question_rephraser(dataframe,user_input,"sentence"))
        # elif "plot" in format.lower():
        #     st.write(question_rephraser(dataframe,user_input,"plot"))
        # else:
        #     print("Can you rephrase your question on which we can able to give you in one of these formats")

        if "table" in format.lower():
            st.write("table output")
            pandas_response = pandas_ai.run(dataframe,prompt="Give me a detailed answer with columns involving with the columns and remove indexes while reporting"+user_input)

            # answer = get_response(pandas_response,user_input)
            st.write(pandas_response)
        elif "sentence" in format.lower():
            st.write("sentence output")
            pandas_response = pandas_ai.run(dataframe,prompt="Give me a detailed answer with columns involving with the columns and remove indexes while reporting"+user_input)

            answer = get_response(pandas_response,user_input)
            st.write(answer)
        
        else:
            print("Can you rephrase your question on which we can able to give you in one of these formats")

        
            # st.session_state.past.append(user_input)
            # st.session_state.generated.append(answer)

            # if st.session_state['generated']:

            #     for i in range(len(st.session_state['generated'])-1, -1, -1):
            #         message(st.session_state["generated"][i],is_table=True, key=str(i))
            #         message(st.session_state['past'][i], is_user=True,is_table=True, key=str(i) + '_user')


    

    


    


