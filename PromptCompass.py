import json
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain  # import LangChain libraries
from langchain.llms import OpenAI  # import OpenAI model
from langchain.chat_models import ChatOpenAI  # import OpenAI chat model
from langchain.callbacks import get_openai_callback  # import OpenAI callbacks
from langchain.prompts import PromptTemplate  # import PromptTemplate


def main():
    # import css tasks and prompts
    with open('selects.json') as f:
        data = json.load(f)

    # title
    st.title("Prompt Compass")
    st.subheader(
        "A Tool for Navigating Prompts for Computational Social Science and Digital Humanities")

    # create input area for task selection
    selections = {}
    tasks_with_names = [task for task in data['tasks'] if task['name']]
    task = st.selectbox('Select a task', tasks_with_names,
                        format_func=lambda x: x['name'] + " - " + x['authors'])
    selections['task'] = task

    # Create input areas for prompts and user input
    input_values = {}

    if task:
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

                    # run LLMChain
                    with get_openai_callback() as cb:
                        llm = OpenAI(temperature=0, model='text-davinci-003',
                                     max_tokens=1024, openai_api_key=os.getenv("OPENAI_API_KEY"))

                        # llm = OpenAIChat(temperature=0, model='gpt-3.5-turbo',
                        #                 max_tokens=1024, openai_api_key=os.getenv("OPENAI_API_KEY"))

                        prompt_template = PromptTemplate(
                            input_variables=["user_input"], template=template)

                        question_chain = LLMChain(
                            llm=llm, prompt=prompt_template)

                        st.success("Input:  "+user_input + "  \n " +
                                   "Output: "+question_chain.run(user_input))
                        st.text(cb)


if __name__ == "__main__":
    main()
