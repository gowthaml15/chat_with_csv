import os 
import streamlit as st
import tempfile
import shutil
import pandas as pd
# import openai
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
import matplotlib
from pandasai import PandasAI
import json

matplotlib.use("TkAgg")
def write_answer(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df_data = {
                    col: [x[i] if isinstance(x, list) else x for x in data['data']]
                    for i, col in enumerate(data['columns'])
                }       
            df = pd.DataFrame(df_data)
            df.set_index("Products", inplace=True)
            st.bar_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

# Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            df.set_index("Products", inplace=True)
            st.line_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")


    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def main():
    
    st.set_page_config(page_title = "Ask you CSV")
    st.header("Ask you CSV")
    open_ai_key = st.text_input("Enter Open AI Key")
    if open_ai_key is not None:
        # _ = load_dotenv(find_dotenv())
        # st.file_uploader("Upload your CSV file",type="csv")
        uploaded_file = st.file_uploader("Upload your CSV file",type="csv")
        
        if uploaded_file is not None:
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
            # download_button = st.download_button(label="Download File", data=user_csv, file_name=user_csv.name)
            # Create a dropdown button
            selected_option = st.selectbox("Select an option for what kind of person you want to answer", ["Data Analyst", "Sales Person","Plot"])

            # Use the selected option
            st.write("You selected:", selected_option)
            question = st.text_input("Enter the question you want to ask about CSV")

            if "Data Analyst" in selected_option:
                query = f""" 
                            As a data analyst, you have been assigned the task of analyzing a dataset and providing insightful findings and recommendations based on your analysis. Summarize your key insights and recommendations in a clear and concise manner, avoiding technical jargon or specific programming code.

                            Query: {question}
                            """
            elif "Sales Person" in selected_option:
                query = f""" 
                            As a salesperson or stakeholder, you have been tasked with providing insights on a specific aspect of your business. Your goal is to share information and analysis without using technical jargon or specific programming code. Write a concise summary that addresses the given prompt in a clear and understandable manner.

                            Query: {question}
                            """
            elif "Plot" in selected_option:
                query = (
                        """
                        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

                        1. If the query requires a table, format your answer like this:
                        {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

                        2. For a bar chart, respond like this:
                        {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

                        3. If a line chart is more appropriate, your reply should look like this:
                        {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

                        Note: We only accommodate two types of charts: "bar" and "line".

                        4. For a plain question that doesn't need a chart or table, your response should be:
                        {"answer": "Your answer goes here"}

                        For example:
                        {"answer": "The Product with the highest Orders is '15143Exfo'"}

                        5. If the answer is not known or available, respond with:
                        {"answer": "I do not know."}

                        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
                        For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

                        Now, let's tackle the query step by step. Here's the query for you to work on: 
                        """
                        + question
                    )
            data = pd.read_csv(destination_path)
            st.write(data.head(3))
            llms = OpenAI(temperature=0,openai_api_key=open_ai_key)
            if question is not None and question != "":
                st.write(f"Your question was:{question}")
                
                if "Plot" in selected_option:
                    # print("---->inside plot")
                    
                    llm = OpenAI(temperature=0.0,openai_api_key=open_ai_key)
                    pandas_ai = PandasAI(llm)
                    # print(data.head())
                    st.write(pandas_ai.run(data,prompt=question))
                    # write_answer(json.loads(str(response)))
                else:
                    agent = create_csv_agent(
                        llms,
                        destination_path,
                        # verbose=True,
                    )
                    response = agent.run(query)
                    st.write(response)



if __name__ == "__main__":
    main()

