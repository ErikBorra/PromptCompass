import json
import streamlit as st
import os
import time
import gc
import pandas as pd
import tiktoken
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
    uploaded_file = None

    # import css tasks and prompts
    with open('prompts.json') as f:
        promptlib = json.load(f)

    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
    st.markdown(hide_default_format, unsafe_allow_html=True)

    # title
    st.title("Prompt Compass")
    st.subheader(
        "A Tool for Navigating Prompts for Computational Social Science and Digital Humanities")
    # Add Link to your repo
    st.markdown(
        '''
        [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/ErikBorra/PromptCompass)
        ''', unsafe_allow_html=True)
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

    with st.expander("Advanced settings"):
        st.markdown(
            """
        **Temperature**: Controls the randomness in the model's responses.
        Lower values (closer to 0.0) make the output more deterministic, while higher values (closer to 2.0) make it more diverse.
        A value of -1 means the parameter will not be specified.
        """
        )
        temperature = st.number_input(
            'Set Temperature', min_value=-1.0, max_value=2.0, value=0.0)

        st.markdown(
            """
        **Top P**: Also known as "nucleus sampling", is an alternative to temperature that can also be used to control the randomness of the model's responses.
        It essentially trims the less likely options in the model's distribution of possible responses. Possible values lie between 0.0 and 1.0. 
        A value of -1 means the parameter will not be specified.
        """
        )
        top_p = st.number_input('Set Top-P', min_value=-
                                1.0, max_value=1.0, value=-1.0)
        st.markdown(
            """
        **Set Maxium Length**: Determines the maximum number of tokens of the **generated** text. A token is approximately four characters word, although this depends on the model.
        A value of -1 means the parameter will not be specified.
        """
        )
        max_new_tokens = st.number_input(
            'Maximum Length', value=-1, min_value=-1, step=1)

    # Check for correct values
    allgood = True
    if not (0 <= temperature <= 2 or temperature == -1):
        st.error(
            "Temperature value must be between 0 and 2. Choose -1 if you want to use the default model value.")
        temperature = -1
        allgood = False
    if not (0 <= top_p <= 1 or top_p == -1):
        st.error(
            "Top P value must be between 0 and 1. Choose -1 if you want to use the default model value.")
        top_p = -1
        allgood = False
    # check if max_new_tokens is at least 1 or -1
    if not (max_new_tokens > 0 or max_new_tokens == -1):
        st.error(
            'Error: Max Tokens must be at least 1. Choose -1 if you want to use the default model value.')
        max_new_tokens = -1
        allgood = False

    # set model kwargs
    model_kwargs = {}
    if 0 <= temperature <= 2:
        model_kwargs['temperature'] = temperature
    if 0 <= top_p <= 1:
        model_kwargs['top_p'] = top_p
    if max_new_tokens > 0:
        model_kwargs['max_tokens'] = max_new_tokens

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
                              ('Text input', 'Upload a CSV'), horizontal=True)

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

        # Determine the output file name
        filename = uploaded_file.name if uploaded_file else 'output.csv'
        base_filename, file_extension = os.path.splitext(filename)
        output_filename = f"{base_filename}_promptcompass{file_extension}"

    # Submit button
    submit_button = st.button('Submit')

    st.write('---')  # Add a horizontal line

    # Process form submission
    if submit_button and allgood:
        if 'user' not in input_values or input_values['user'] is None:
            st.error("No user input provided")

        else:
            with st.spinner(text="In progress..."):

                start_time = time.time()
                st.write("Start time: " +
                         time.strftime("%H:%M:%S", time.localtime()))

                # if model has changed, free up memory of the previous one
                if input_values['model']['name'] != st.session_state.get('previous_model'):
                    pipe = None
                    torch.cuda.empty_cache()
                    gc.collect()  # garbage collection
                    st.write('Changing model from ' +
                             st.session_state['previous_model'] + ' to ' + input_values['model']['name'] + '...')
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

                    # make sure users don't forget the user input variable
                    if "{user_input}" not in template:
                        template = template + "\n\n{user_input}"

                    # fill prompt template
                    prompt_template = PromptTemplate(
                        input_variables=["user_input"], template=template)

                    # loop over user values in prompt
                    for key, user_input in enumerate(input_values['user']):

                        num_tokens = None

                        user_input = str(user_input).strip()
                        if user_input == "" or user_input == "nan":
                            continue

                        # set up and run the model
                        model_id = input_values['model']['name']
                        if model_id in ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'text-davinci-003', 'text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001', 'gpt-4']:
                            if open_ai_key is None or open_ai_key == "":
                                st.error("Please provide an Open AI API Key")
                                exit(1)
                            with get_openai_callback() as cb:
                                if model_id in ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4']:
                                    llm = ChatOpenAI(
                                        model=model_id, openai_api_key=open_ai_key, **model_kwargs)
                                else:
                                    llm = OpenAI(
                                        model=model_id, openai_api_key=open_ai_key, **model_kwargs)

                                llm_chain = LLMChain(
                                    llm=llm, prompt=prompt_template)

                                output = llm_chain.run(user_input)

                                encoding = tiktoken.encoding_for_model(
                                    model_id)
                                num_tokens = len(encoding.encode(
                                    prompt_template.format(user_input=user_input)))

                                st.success("Input:  " + user_input + "  \n\n " +
                                           "Input tokens (incl. prompt): " + str(num_tokens) + "  \n\n " +
                                           "Output: " + output)
                                st.text(cb)
                        elif model_id in ['tiiuae/falcon-7b', 'mosaicml/mpt-7b']:
                            if pipe == None:
                                tokenizer = AutoTokenizer.from_pretrained(
                                    model_id)

                                pipe = pipeline(
                                    "text-generation",
                                    model=model_id,
                                    tokenizer=tokenizer,
                                    torch_dtype=torch.bfloat16,
                                    trust_remote_code=True,
                                    device_map="auto",
                                    do_sample=True,
                                    top_k=10,
                                    num_return_sequences=1,
                                    eos_token_id=tokenizer.eos_token_id,
                                )

                                local_llm = HuggingFacePipeline(
                                    pipeline=pipe, model_kwargs=model_kwargs)
                                st.write('Model loaded')

                            llm_chain = LLMChain(
                                llm=local_llm, prompt=prompt_template)

                            output = llm_chain.run(user_input)
                            st.success("Input:  " + user_input + "  \n " +
                                       "Output: " + output)
                        elif model_id in ['google/flan-t5-large', 'google/flan-t5-xl', 'tiiuae/falcon-7b-instruct']:
                            if pipe is None:
                                tokenizer = AutoTokenizer.from_pretrained(
                                    model_id)

                                if model_id in ['google/flan-t5-large', 'google/flan-t5-xl']:
                                    model = AutoModelForSeq2SeqLM.from_pretrained(
                                        model_id, load_in_8bit=False, device_map='auto')
                                    pipe = pipeline(
                                        "text2text-generation",
                                        model=model,
                                        tokenizer=tokenizer,
                                        device_map="auto"
                                    )
                                elif model_id == 'tiiuae/falcon-7b-instruct':
                                    pipe = pipeline(
                                        "text-generation",
                                        model=model_id,
                                        tokenizer=tokenizer,
                                        torch_dtype=torch.bfloat16,
                                        trust_remote_code=True,
                                        device_map="auto",
                                        do_sample=True,
                                        top_k=10,
                                        num_return_sequences=1,
                                        eos_token_id=tokenizer.eos_token_id
                                    )

                                local_llm = HuggingFacePipeline(
                                    pipeline=pipe, model_kwargs=model_kwargs)
                                st.write('Model loaded')

                            llm_chain = LLMChain(
                                llm=local_llm, prompt=prompt_template)

                            output = llm_chain.run(user_input)
                            num_tokens = len(tokenizer.tokenize(
                                prompt_template.format(user_input=user_input)))
                            st.success("Input:  " + user_input + "  \n\n " +
                                       "Input tokens (incl. prompt): " + str(num_tokens) + "  \n\n " +
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
                                    # temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                                    # top_p=0.15,  # select from top tokens whose probability add up to 15%
                                    # select from top 0 tokens (because zero, relies on top_p)
                                    top_k=0,
                                    # max_new_tokens=64,  # mex number of tokens to generate in the output
                                    repetition_penalty=1.1  # without this output begins repeating
                                )
                                local_llm = HuggingFacePipeline(
                                    pipeline=pipe, model_kwargs=model_kwargs)

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
                        data.loc[key, 'llm'] = model_id
                        data.loc[key, 'prompt timestamp'] = time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime())
                        data.loc[key, 'prompt name'] = task['name']
                        data.loc[key, 'prompt authors'] = task['authors']
                        data.loc[key, '# input tokens'] = str(int(num_tokens))
                        data.loc[key, 'prompt'] = template
                        if "temperature" in model_kwargs:
                            data.loc[key,
                                     'temperature'] = model_kwargs['temperature']
                        if "top_p" in model_kwargs:
                            data.loc[key, 'top_p'] = model_kwargs['top_p']
                        if "max_tokens" in model_kwargs:
                            data.loc[key,
                                     'max_tokens'] = int(model_kwargs['max_tokens'])

                    # make output available as csv
                    csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        csv,
                        output_filename,
                        "text/csv",
                        key='download-csv'
                    )

                end_time = time.time()
                elapsed_time = end_time - start_time
                st.write("End time: " +
                         time.strftime("%H:%M:%S", time.localtime()))
                st.write("Elapsed time: " +
                         str(round(elapsed_time, 2)) + " seconds")


if __name__ == "__main__":
    main()
