import datetime



def template_based_approach(question,df):
    today_date = datetime.date.today()
    num_rows = len(df)
    num_columns = len(df.columns)
    df_head = df.head()
    suffix = "\n\nCode:\n"
    steps = ""
    db = {
            "week end":"""
                    We need to provide total sales/revenue a 7 day period per the date presented
                        Steps:
                        -Data source = Orders
                        -Week ended = the date presented (e.g. 25 Jul) minus 6 days. 
                        -Identify date range , e.g. 25 Jul - 6  days= 19 Jul, hence date range is 19 Jul to 25 Jul
                        -Sum “Sales Price” if “Purchase Date” is between date range
                            """
            }
    for i,(keys,values) in enumerate(db.items()):
        if keys in question:
            steps = values
    
    if len(steps)>=2:
        code_generation_2 = f"""Today is {today_date}.
        You are provided with a pandas dataframe (df) with {num_rows} rows and {num_columns} columns and steps to generate the code.
        This is the metadata of the dataframe:
        {df_head}.

        Steps to be followed:
        {steps}

        Using the provided dataframe and steps, df, return the python function code with neccessary import modules in the below format: 
        def ploting(dataframe) calling the function and return as response and store that in result variable
        Your answer to the following question:
        {question}
        
        {suffix}
        """ 
        return code_generation_2

    return steps
    