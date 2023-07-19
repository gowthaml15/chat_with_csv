
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
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain,LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory,ConversationSummaryBufferMemory
from template import template_based_approach
import matplotlib
import io

matplotlib.use("TkAgg")

def custom_exec(code, namespace):
    # Execute the code using built-in exec with the provided namespace
    exec(code, namespace)


def template_approach(user_input,dataframe):
    code_generation = template_based_approach(user_input,dataframe)
    response =""
    flag = False
    # openai.api_key  = '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'
    if len(code_generation) > 2:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", # this is "ChatGPT" $0.002 per 1k tokens
        messages=[{"role": "user", "content": f"{code_generation}"}]
        )
        string = completion["choices"][0]["message"]["content"]

        # Create the namespace with the required variables
        namespace = {"df": dataframe}

        custom_exec(string, namespace)
        print(string)
        # Access the total_revenue value
        response = namespace.get("result")
        if len(str(response))>5:
            flag = True
    return response, flag

def get_question():
    return st.text_input("Talk to the bot",key="input_text")

def question_analyiser(question):
    llm = OpenAI(temperature=0.0)
    prompt = """Your task is to classify which type of question is that from the mentioned class,The class are 'sentence' and 'table' and 'plot'. The question will be given by the user. {question} answer in one word like 'table' or 'sentence' or 'plot without any spaces"""
    prompt = PromptTemplate(template=prompt,input_variables=["question"])
    llmchain = LLMChain(llm=llm,prompt=prompt)
    response = llmchain.run({"question":question})
    return response

def get_columns (question,dataframe):
    llm = OpenAI(temperature=0.0)
    
    prompt = """ Your task is to find the appropriate columns from the dataframe that is related to the question 
    question : {question}
    dataframe : {df}
    """

    code_generation_prompt = PromptTemplate(template=prompt,input_variables=["question","df"])
    llmchain = LLMChain(llm=llm,prompt=code_generation_prompt)
    columns_response = llmchain.run({"question":question,"df":dataframe})
    return columns_response

def get_response(llm = None,generated="",question="",format=""):
    
    llm = OpenAI(temperature=0.0)
    prompt = """ 
            As a data analyst bit you have to report the generated answer in a detail manner, and give if the answer is in money realted with no more than 2 precision and is this format {format}:
            {generated}
            For the question:
            {question}
            """
    format_prompt = PromptTemplate(template=prompt,input_variables=["format","generated","question"])
    llmchain = LLMChain(llm=llm,prompt=format_prompt)
    response = llmchain.run({"question":question,"format":format,"generated":generated})
    
    return response

def explainer(ll=None,question="", answer="", columns=""):
    llm = OpenAI(temperature=0.0)

    prompt = """
            Your task is to explain the answer like a data analyst, on how the answer is got with use of the columns, 
            question : {question}
            columns : {columns}
            answer : {answer}
            Give me the explaination for the answer
            """
    format_prompt = PromptTemplate(template=prompt,input_variables=["columns","answer","question"])
    llmchain = LLMChain(llm=llm,prompt=format_prompt)
    response = llmchain.run({"question":question,"answer":answer,"columns":columns})
    return response

def main():
    st.set_page_config(page_title = "Ask you CSV")
    st.header("Ask you CSV")
    open_ai_key = st.text_input("Enter Open AI Key")
    if open_ai_key:
        openai.api_key = open_ai_key

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
                dataframe.append(dj)
                st.write(dj.head(3))

            if 'generated' not in st.session_state:
                st.session_state['generated'] = []
            if 'past' not in st.session_state:
                st.session_state['past'] = []
            
            user_input = get_question()
            if user_input:
                response, flag = template_approach(user_input,dataframe[0])

                if flag:
                    format_response = question_analyiser(user_input)
                    columns = get_columns(user_input,dataframe[0])
                    answer = get_response(generated=response,question=user_input,format=format_response)
                    st.write(answer)
                    st.write(explainer(question=user_input,answer=answer,columns=columns))

                else:
                    pandas_response = pandas_ai.run(dataframe,prompt=user_input)
                    if len(pandas_response) >=2:
                        format_response = question_analyiser(user_input)
                        columns = get_columns(user_input,dataframe)
                        answer = get_response(generated=pandas_response,question=user_input,format=format_response)
                        st.write(answer)
                        st.write(explainer(question=user_input,answer=answer,columns=columns))

                    else:
                        st.write("Can you rephrase the question so that I can provide you the answer")



if __name__ == "__main__":
    main()
    

    


    


