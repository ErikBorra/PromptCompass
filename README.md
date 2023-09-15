# PromptCompass

[![DOI](https://zenodo.org/badge/649855474.svg)](https://zenodo.org/badge/latestdoi/649855474)
[![Requires Python 3.9](https://img.shields.io/badge/py-v3.9-blue)](https://www.python.org/)

Prompt Compass is a tool designed to leverage Language Learning Models (LLMs) in digital research tasks. It accomplishes this by offering access to diverse LLMs, supplying a library of prompts for digital research, and enabling users to apply these prompts to a series of inputs.

The tool offers two categories of LLMs: local LLMs and platform APIs. Local LLMs can be used to provide stable and reproducible results, facilitate in-depth analysis, and support a robust interpretation. These LLMs are optimized to run on a GPU with 24GB of RAM. Platform APIs, however, unlock the power of more advanced and sophisticated LLMs.

Our library of prompts comprises prompts extracted from academic literature and other research, each of which is linked to its respective source. This feature enables you to examine the objectives and effectiveness of each prompt. It is also possible to create custom prompts.

The tool accepts user inputs either as text lines or CSV files, to be processed with the chosen LLM and prompt, with each line treated separately.

The results are exported in a CSV file, which includes columns specifying the LLM used, its parameters, the prompt employed, and the corresponding output.

## Install
`pip install git+https://github.com/huggingface/transformers`
`pip install -r requirements.txt`

`cp .env.example .env`
And enter your open_ai_key

To use the llama-2 models, you first need to apply for access to the llama-2 models via e.g. https://huggingface.co/meta-llama/Llama-2-7b-chat-hf Once accepted, get a hugging face auth token https://huggingface.co/settings/tokens and then run `huggingface-cli login` on the command line, filling in the generated token.

## Run
`streamlit run PromptCompass.py`

Then navigate your browser to the URL shown

## FAQ

- Creating Custom Prompts
    - If you wish to use a customized prompt, please select 'custom - instruct - user' and replace it with your desired prompt. Ensure to include {user_input} in your prompt, this is where your input will be inserted.
    - It's important to remember to incorporate {user_input} within your custom prompt. This will act as a placeholder that will be filled by each line of your input as the script iterates over them.
- Importing CSV Files
    - If you have data in a CSV format, you can easily upload it for analysis. Following the upload, you will have the option to choose a specific column for analysis. The script will then navigate through each row, replacing the {user_input} in your text with the corresponding cell from the selected column in each row.
- Required VRAM 
    - You can use https://huggingface.co/spaces/hf-accelerate/model-memory-usage to get a good idea of how much VRAM you need for a specific model and with what precision (i.e. whether you can run it on your own machine).
- Errors
    - If you get an error like the following `pandas.errors.ParserError: Error tokenizing data. C error: Expected 1 fields in line 13, saw 3` your CSV probably does not have a header row.
    - If you get an error like the following `Retrying langchain.chat_models.openai.ChatOpenAI.completion_with_retry.<locals>._completion_with_retry in 16.0 seconds as it raised RateLimitError: Rate limit reached for default-gpt-3.5-turbo-16k in organization org-WlgvnCb1Wg24MDcyDC7Y58bK on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at` `[help.openai.com](http://help.openai.com/)` `if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit` `[https://platform.openai.com/account/billing](https://platform.openai.com/account/billing)` `to add a payment method..` consider adding a payment method to your openai account
    - If you encounter an error message stating 'gpt-4' model not found, it's recommended to verify that you've completed at least one successful transaction through the OpenAI developer platform. This error may arise if a payment has not been successfully processed there.