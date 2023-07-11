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
import torch  # import torch
# pip install git+https://github.com/huggingface/transformers
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


def main():
    load_dotenv(".env")

    pipe = None
    open_ai_key = None

    # import css tasks and prompts
    with open('prompts.json') as f:
        promptlib = json.load(f)

    # title
    st.title("Prompt Compass")
    st.subheader(
        "A Tool for Navigating Prompts for Computational Social Science and Digital Humanities")

    # load available models
    model_with_names = [
        model for model in promptlib['models'] if model['name']]

    # create input area for model selection
    input_values = {}

    input_values['model'] = st.selectbox('Select a model', model_with_names,
                                         format_func=lambda x: x['name'])

    # If there is no previous state, set the default model as the first model
    if not st.session_state.get('previous_model'):
        st.session_state['previous_model'] = model_with_names[0]['name']

    st.caption("Model info: [" + input_values['model']
               ['name']+"]("+input_values['model']['resource']+")")

    # ask for open ai key if no key is set in .env
    if input_values['model']['resource'] in ["https://platform.openai.com/docs/models/gpt-3-5", "https://platform.openai.com/docs/models/gpt-4"]:
        # Load the OpenAI API key from the environment variable
        if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
            open_ai_key = st.text_input("Open AI API Key", "")
        else:
            open_ai_key = os.getenv("OPENAI_API_KEY")

    # create input area for task selection
    tasks_with_names = [task for task in promptlib['tasks'] if task['name']]
    task = st.selectbox('Select a task', tasks_with_names,
                        format_func=lambda x: x['name'] + " - " + x['authors'])

    # Create input areas for prompts and user input

    if task:

        # concatenate all strings from prompt array
        prompt = '\n'.join(task['prompt'])

        # create input area for prompt
        input_values['prompt'] = st.text_area(
            "Inspect, and possibly modify, the prompt by ["+task['authors']+"]("+task['paper']+")", prompt, height=200)

        # allow the user to select the input type
        input_type = st.radio("Choose input type:",
                              ('Text input', 'Upload a CSV'))

        if input_type == 'Text input':
            # create input area for user input
            input_values['user'] = st.text_area(
                "Input to be analyzed with the prompt (one thing per line):",
                "this user is happy\none user is just a user\nthe other user is a lier")
            # if the user's input is not a list (e.g. a string), then split it by newlines
            if isinstance(input_values['user'], str):
                input_values['user'] = input_values['user'].split('\n')
            data = pd.DataFrame(input_values['user'], columns=['user_input'])
        else:
            # upload CSV
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:
                # convert the uploaded file to a dataframe
                data = pd.read_csv(uploaded_file)

                # ask user to select a column
                column_to_extract = st.selectbox(
                    'Choose a column to apply the prompt on:', data.columns)

                # process the selected column from the dataframe
                input_values['user'] = data[column_to_extract].tolist()

    # Submit button
    submit_button = st.button('Submit')
    st.write('---')  # Add a horizontal line

    # Process form submission
    if submit_button:
        with st.spinner(text="In progress..."):

            start_time = time.time()
            st.write("Start time: " +
                     time.strftime("%H:%M:%S", time.localtime()))

            # if model has changed, free up memory of the previous one
            if input_values['model']['name'] != st.session_state.get('previous_model'):
                pipe = None
                torch.cuda.empty_cache()
                # st.write('Changing model from ' + st.session_state['previous_model'] + ' to ' + input_values['model']['name'] + '...')
                st.session_state['previous_model'] = input_values['model']['name']

            if input_values['prompt'] and input_values['user']:

                # create prompt template
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

                # loop over user values in prompt
                for key, user_input in enumerate(input_values['user']):

                    user_input = str(user_input).strip()
                    if user_input == "" or user_input == "nan":
                        continue

                    # set up and run the model
                    model_id = input_values['model']['name']
                    if model_id in ['gpt-3.5-turbo', 'text-davinci-003', 'text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001', 'gpt-4']:
                        if open_ai_key is None or open_ai_key == "":
                            st.error("Please provide an Open AI API Key")
                            exit(1)
                        with get_openai_callback() as cb:
                            if model_id in ['gpt-3.5-turbo', 'gpt-4']:
                                llm = ChatOpenAI(temperature=0, model=model_id,
                                                 max_tokens=1024, openai_api_key=open_ai_key)
                            else:
                                llm = OpenAI(temperature=0, model=model_id,
                                             max_tokens=1024, openai_api_key=open_ai_key)

                            llm_chain = LLMChain(
                                llm=llm, prompt=prompt_template)

                            output = llm_chain.run(user_input)
                            st.success("Input:  " + user_input + "  \n " +
                                       "Output: " + output)
                            st.text(cb)
                    elif model_id in ['tiiuae/falcon-7b', 'mosaicml/mpt-7b']:
                        if pipe == None:
                            tokenizer = AutoTokenizer.from_pretrained(model_id)

                            pipe = pipeline(
                                "text-generation",
                                model=model_id,
                                tokenizer=tokenizer,
                                torch_dtype=torch.bfloat16,
                                trust_remote_code=True,
                                device_map="auto",
                                max_length=200,
                                do_sample=True,
                                top_k=10,
                                num_return_sequences=1,
                                eos_token_id=tokenizer.eos_token_id,
                            )

                            local_llm = HuggingFacePipeline(
                                pipeline=pipe, model_kwargs={'temperature': 0.0})
                            st.write('Model loaded')

                        llm_chain = LLMChain(
                            llm=local_llm, prompt=prompt_template)

                        output = llm_chain.run(user_input)
                        st.success("Input:  " + user_input + "  \n " +
                                   "Output: " + output)
                    elif model_id in ['google/flan-t5-large', 'tiiuae/falcon-7b-instruct']:
                        if pipe is None:
                            tokenizer = AutoTokenizer.from_pretrained(model_id)

                            if model_id == 'google/flan-t5-large':
                                model = AutoModelForSeq2SeqLM.from_pretrained(
                                    model_id, load_in_8bit=True, device_map='auto')
                                pipe = pipeline(
                                    "text2text-generation",
                                    model=model,
                                    tokenizer=tokenizer,
                                    device_map="auto",
                                    max_length=200
                                )
                            elif model_id == 'tiiuae/falcon-7b-instruct':
                                pipe = pipeline(
                                    "text-generation",
                                    model=model_id,
                                    tokenizer=tokenizer,
                                    torch_dtype=torch.bfloat16,
                                    trust_remote_code=True,
                                    device_map="auto",
                                    max_length=200,
                                    do_sample=True,
                                    top_k=10,
                                    num_return_sequences=1,
                                    eos_token_id=tokenizer.eos_token_id
                                )

                            local_llm = HuggingFacePipeline(
                                pipeline=pipe, model_kwargs={'temperature': 0.0})
                            st.write('Model loaded')

                        llm_chain = LLMChain(
                            llm=local_llm, prompt=prompt_template)

                        output = llm_chain.run(user_input)
                        st.success("Input:  " + user_input + "  \n " +
                                   "Output: " + output)
                    elif model_id == "mosaicml/mpt-7b-instruct":

                        if pipe is None:
                            # st.write('Loading model '+model_id)

                            model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                trust_remote_code=True,
                                torch_dtype=torch.bfloat16,
                                max_seq_len=2048,
                                device_map="auto"
                            )

                            # MPT-7B model was trained using the EleutherAI/gpt-neox-20b tokenizer
                            tokenizer = AutoTokenizer.from_pretrained(
                                "EleutherAI/gpt-neox-20b")

                            # mtp-7b is trained to add "<|endoftext|>" at the end of generations
                            stop_token_ids = tokenizer.convert_tokens_to_ids(
                                ["<|endoftext|>"])
                            # define custom stopping criteria object

                            class StopOnTokens(StoppingCriteria):
                                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                                    for stop_id in stop_token_ids:
                                        if input_ids[0][-1] == stop_id:
                                            return True
                                    return False
                            stopping_criteria = StoppingCriteriaList(
                                [StopOnTokens()])

                            pipe = pipeline(
                                task='text-generation',
                                model=model,
                                tokenizer=tokenizer,
                                return_full_text=True,  # langchain expects the full text
                                # we pass model parameters here too
                                stopping_criteria=stopping_criteria,  # without this model will ramble
                                temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                                top_p=0.15,  # select from top tokens whose probability add up to 15%
                                # select from top 0 tokens (because zero, relies on top_p)
                                top_k=0,
                                max_new_tokens=64,  # mex number of tokens to generate in the output
                                repetition_penalty=1.1  # without this output begins repeating
                            )
                            local_llm = HuggingFacePipeline(pipeline=pipe)

                            st.write("model loaded")

                        llm_chain = LLMChain(
                            llm=local_llm, prompt=prompt_template)

                        output = llm_chain.run(user_input)
                        st.success("Input:  " + user_input + "  \n " +
                                   "Output: " + output)
                    else:
                        st.error("Model not found")
                        exit(1)

                    # add output to dataframe
                    data.loc[key, 'output'] = output
                    data.loc[key, 'model'] = model_id
                    data.loc[key, 'task'] = task['name']
                    data.loc[key, 'authors'] = task['authors']
                    data.loc[key, 'prompt'] = template
                    data.loc[key, 'timestamp'] = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime())

                # make output available as csv
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    csv,
                    "output.csv",
                    "text/csv",
                    key='download-csv'
                )

            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write("End time: " + time.strftime("%H:%M:%S", time.localtime()))
            st.write("Elapsed time: " +
                     str(round(elapsed_time, 2)) + " seconds")


if __name__ == "__main__":
    main()
