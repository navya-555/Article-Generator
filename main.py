import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

loaded_llm= HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    #repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    token=os.getenv("hf_token"),
    timeout=500,
    max_new_tokens=700)

llm = ChatHuggingFace(llm=loaded_llm)

template="""
            Write an article for {article_style} on the topic {input_text}
            within {no_words} and the tone should be {tone}.
        """

prompt=PromptTemplate(input_variables=['input_text','no_words','article_style','tone'],
                        template=template)


def get_response(input_text,no_words,article_style,tone):
    
    response=llm.invoke(prompt.format(input_text=input_text,no_words=no_words,article_style=article_style,tone=tone))

    print(response)
    return response


st.set_page_config(page_title="Generate Articles",
                    page_icon='bot.png',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Articles 🤖")

input_text=st.text_input("Enter the topic for article")

col1,col2,col3=st.columns([5,5,5])

with col1:
    no_words=st.text_input('No of words')
with col2:
    article_style=st.selectbox('Article is for',('Newspaper','Commercial Magazine','School Magazine','Social Media','Organization'))
with col3:
    tone=st.selectbox('Tone should be',('Formal','Informal'))

submit=st.button("Generate")

if submit:
    st.write(get_response(input_text,no_words,article_style,tone).content)
