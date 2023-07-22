from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

import streamlit as st

import os
from apikey import apikey 
os.environ['OPENAI_API_KEY'] = apikey

st.title(" 🧜🏽‍♀️🧜🏽‍♀️🧜🏽‍♀️ AI GPT Creator 🧜🏽‍♀️🧜🏽‍♀️🧜🏽‍♀️")

prompt = st.text_input('Plug in your prompt here')

title_template = PromptTemplate(
    input_variables = ['topic'],
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template='write me a youtube video script based on this title TITLE: {title} \
    while leveraging this wikipedia research:{wikipedia_research}'
)

#memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title','script'], verbose=True)
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    # response = sequential_chain({'topic':prompt})

    st.write(title)
    st.write(script)
    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('script History'):
        st.info(script_memory.buffer)
    
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)