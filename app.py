import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain, SequentialChain




os.environ['OPENAI_API_KEY'] = apikey
# app work
st.title('🦜🔗 GPT')
prompt = st.text_input('plug in your prompt here')

# prompt template
title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a debate  title about {topic}'
)
script_template = PromptTemplate(
    input_variables=['title'],
    template='write me a debate based on this title TITLE: {title}'
)
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script')
sequential_chain = SequentialChain(chains=[title_chain, script_chain], verbose=True,
                                   input_variables=['topic'], output_variables=['title', 'script'])
if prompt:
    response = sequential_chain({'topic': prompt})
    st.write(response['script'])
    st.write(response['title'])
