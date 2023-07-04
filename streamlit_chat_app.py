import os 
import streamlit as st
import tempfile
import shutil
import pandas as pd
import json
import matplotlib
# import openai

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
from pandasai import PandasAI
from streamlit_chat import message
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory


def get_llm(key):
    llms = OpenAI(temperature=0,openai_api_key=key)
    return llms

def generate_answer(llms,question,destination_path,history):
    data = pd.read_csv(destination_path)
    st.write(data.head(3))
    prompt = """The following is a friendly conversation between a human and an AI. The AI is a data analyst, you have been assigned the task of analyzing a dataset and providing insightful findings and recommendations based on your analysis. Summarize your key insights and recommendations in a clear and concise manner, avoiding technical jargon or specific programming code.

    If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.
    Conversation:
    Human: {question}
    AI:"""


    agent = create_csv_agent(
        llms,
        destination_path,
        prompt=prompt,
        verbose=True,
    )
    response = agent.run(question)

    return response

def get_question():
    return st.text_input("Talk to the bot",key="input_text")

def get_file(uploaded_file):

    file_path = os.path.abspath(uploaded_file.name)
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded file to the temporary directory
    file_path = temp_dir + "/" + uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Copy the file to the code directory
    output_dir = "./data"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    destination_path = os.path.join(output_dir, uploaded_file.name)
    shutil.copy(file_path, destination_path)

    # Display a success message
    st.success(f"File saved to: {destination_path}")

    return destination_path
    

def main():
    st.set_page_config(page_title = "Chat with your CSV")
    st.header("Chat with your CSV")

    key = st.text_input("Enter Open AI Key")
    if key:
        llms = get_llm(key)
        uploaded_file = st.file_uploader("Upload your CSV file",type="csv")
        if uploaded_file:
            destination_path = get_file(uploaded_file)

            if 'generated' not in st.session_state:
                st.session_state['generated'] = []
            if 'past' not in st.session_state:
                st.session_state['past'] = []
            
            user_input = get_question()

            if user_input:
                output = generate_answer(llms,user_input,destination_path,st.session_state.generated)

                st.session_state.past.append(user_input)
                st.session_state.generated.append(output)

                if st.session_state['generated']:

                    for i in range(len(st.session_state['generated'])-1, -1, -1):
                        message(st.session_state["generated"][i], key=str(i))
                        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

                
                # print(st.session_state.past,st.session_state.generated)


if __name__ == "__main__":
    main()