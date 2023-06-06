import json
import streamlit as st
import os
import time
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import LLMChain  # import LangChain libraries
from langchain.llms import OpenAI  # import OpenAI model
from langchain.chat_models import ChatOpenAI  # import OpenAI chat model
from langchain.callbacks import get_openai_callback  # import OpenAI callbacks
from langchain.prompts import PromptTemplate  # import PromptTemplate
from langchain.llms import HuggingFacePipeline  # import HuggingFacePipeline
import torch # import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM # import transformers


def main():
    # import css tasks and prompts
    with open('prompts.json') as f:
        promptlib = json.load(f)

    # title
    st.title("Prompt Compass")
    st.subheader(
        "A Tool for Navigating Prompts for Computational Social Science and Digital Humanities")

    # create input area for task selection
    tasks_with_names = [task for task in promptlib['tasks'] if task['name']]
    task = st.selectbox('Select a task', tasks_with_names,
                        format_func=lambda x: x['name'] + " - " + x['authors'])

    # Create input areas for prompts and user input
    input_values = {}

    if task:
        # create input area for model selection
        model_with_names = [model for model in promptlib['models'] if model['name']]
        input_values['model'] = st.selectbox('Select a model', model_with_names, 
                        format_func=lambda x: x['name'])

        # concatenate all strings from prompt array
        prompt = '\n'.join(task['prompt'])

        # create input area for prompt
        input_values['prompt'] = st.text_area(
            "Inspect, and possibly modify, the prompt by ["+task['authors']+"]("+task['paper']+")", prompt, height=200)

        # create input area for user input
        input_values['user'] = st.text_area(
            "User input (one thing to be analyzed per line):", "this user is happy\none user is just a user\nthe other user is a lier")

    # Submit button
    submit_button = st.button('Submit')
    st.write('---')  # Add a horizontal line
    st.write("Using model " + input_values['model']['name'])

    # Process form submission
    if submit_button:
        with st.spinner(text="In progress..."):
            load_dotenv()

            start_time = time.time()
            st.write("Start time: " + time.strftime("%H:%M:%S", time.localtime()))

            # Load the OpenAI API key from the environment variable
            if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
                st.error("OPENAI_API_KEY is not set")
                exit(1)
            
            

            
            if input_values['prompt'] and input_values['user']:

                # create dataframe for output
                df = pd.DataFrame(columns=['input', 'output','model','task','authors','prompt'])

                # split user input into array
                input_values['user'] = input_values['user'].split('\n')

                # loop over user values in prompt
                for user_input in input_values['user']:
                    
                    # add location of user input to prompt
                    if task['location_of_input'] == 'before':
                        template = "{user_input}" + \
                            "\n\n" + input_values['prompt']
                    elif task['location_of_input'] == 'after':
                        template = input_values['prompt'] + \
                            "\n\n" + "{user_input}"
                    else:
                        template = input_values['prompt']

                    # fill prompt template
                    prompt_template = PromptTemplate(
                                input_variables=["user_input"], template=template)
                    
                    # set up and run the model
                    model_id = input_values['model']['name']
                    if model_id == 'text-davinci-003':
                        with get_openai_callback() as cb:
                            llm = OpenAI(temperature=0, model=model_id,
                                        max_tokens=1024, openai_api_key=os.getenv("OPENAI_API_KEY"))
                            
                            llm_chain = LLMChain(
                                llm=llm, prompt=prompt_template)
                            
                            output = llm_chain.run(user_input)
                            st.success("Input:  " + user_input + "  \n " +
                                    "Output: " + output)
                            st.text(cb)
                    elif model_id in ['google/flan-t5-large']:
                        torch.cuda.empty_cache()

                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                        
                        if model_id == 'google/flan-t5-large':
                            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')
                            pipe = pipeline(
                                "text2text-generation",
                                model=model, 
                                tokenizer=tokenizer, 
                                max_length=100
                            )
                        
                        local_llm = HuggingFacePipeline(pipeline=pipe)
                        
                        llm_chain = LLMChain(
                            llm=local_llm, prompt=prompt_template)
                        
                        output = llm_chain.run(user_input)
                        st.success("Input:  " + user_input + "  \n " +
                                "Output: " + output)
                    
                    else:
                        st.error("Model not found")
                        exit(1)

                    # add output to dataframe
                    new_row = {
                                'input': user_input,
                                'output': output,
                                'model': model_id,
                                'task': task['name'], 
                                'authors': task['authors'],
                                'prompt': template
                            }
                    df.loc[len(df.index)] = new_row

                # make output available as csv
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Press to Download",
                    csv,
                    "output.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            end_time = time.time()  
            elapsed_time = end_time - start_time
            st.write("End time: " + time.strftime("%H:%M:%S", time.localtime()))
            st.write("Elapsed time: " + str(round(elapsed_time,2)) + " seconds")

if __name__ == "__main__":
    main()
