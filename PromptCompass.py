import json
import streamlit as st
import os
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
    with open('selects.json') as f:
        data = json.load(f)

    # title
    st.title("Prompt Compass")
    st.subheader(
        "A Tool for Navigating Prompts for Computational Social Science and Digital Humanities")

    # create input area for task selection
    tasks_with_names = [task for task in data['tasks'] if task['name']]
    task = st.selectbox('Select a task', tasks_with_names,
                        format_func=lambda x: x['name'] + " - " + x['authors'])

    # Create input areas for prompts and user input
    input_values = {}

    if task:
        # create input area for model selection
        model_with_names = [model for model in data['models'] if model['name']]
        input_values['model'] = st.selectbox('Select a model', model_with_names, 
                        format_func=lambda x: x['name'])

        # concatenate all strings from prompt array
        prompt = '\n'.join(task['prompt'])

        # create input area for prompt
        input_values['prompt'] = st.text_area(
            "Inspect, and possibly modify, the prompt by ["+task['authors']+"]("+task['doi']+")", prompt, height=200)

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

            # Load the OpenAI API key from the environment variable
            if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
                st.error("OPENAI_API_KEY is not set")
                exit(1)

            # loop over user values in prompt
            if input_values['prompt'] and input_values['user']:
                # split user input into array
                input_values['user'] = input_values['user'].split('\n')

                # add location of user input
                for user_input in input_values['user']:
                    if task['location_of_input'] == 'before':
                        template = "{user_input}" + \
                            "\n\n" + input_values['prompt']
                    elif task['location_of_input'] == 'after':
                        template = input_values['prompt'] + \
                            "\n\n" + "{user_input}"
                    else:
                        template = input_values['prompt']

                    # create prompt template
                    prompt_template = PromptTemplate(
                                input_variables=["user_input"], template=template)
                    
                    # run the model
                    if input_values['model']['name'] == 'text-davinci-003':
                        with get_openai_callback() as cb:
                            llm = OpenAI(temperature=0, model=input_values['model']['name'],
                                        max_tokens=1024, openai_api_key=os.getenv("OPENAI_API_KEY"))
                            
                            question_chain = LLMChain(
                                llm=llm, prompt=prompt_template)
                            st.success("Input:  "+user_input + "  \n " +
                                    "Output: "+question_chain.run(user_input))
                            st.text(cb)
                    elif input_values['model']['name'] == 'google/flan-t5-large':
                        model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')

                        pipe = pipeline(
                            "text2text-generation",
                            model=model, 
                            tokenizer=tokenizer, 
                            max_length=100
                        )

                        local_llm = HuggingFacePipeline(pipeline=pipe)
                        llm_chain = LLMChain(
                            llm=local_llm, prompt=prompt_template, verbose=True)
                        
                        st.success("Input:  "+user_input + "  \n " +
                                "Output: "+llm_chain.run(user_input))

if __name__ == "__main__":
    main()
