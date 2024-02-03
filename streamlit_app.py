import streamlit as st
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

job_description = '''
The project is to analyze the nonces submitted by different types of hardware and create an identifying signature for each of the hardware.
An expert ML person is required.
Person should be an expert programmer and be able to pull data, do the anlaysis, build the models and create a usable dashboard or excel for the end-user.
Good knowledge of blockchain and proof-of-work mining will be desirable.
Good knowledge of c++ or firmware is desirable.
Person should be available full-time and be flexible for calls over weekends and late evenings
'''

def get_resume(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = [Document(page_content=x) for x in text_splitter.split_text(text)]

        with get_openai_callback() as cb:
            resume_db = FAISS.from_documents(docs, embeddings)
            print(f'Resume embeding: {cb}')
        return resume_db

            
def get_job_description(job_description):
    if job_description:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = [Document(page_content=x) for x in text_splitter.split_text(job_description)]
        with get_openai_callback() as cb:
            job_description_db = FAISS.from_documents(docs, embeddings)
            print(f'job embeding: {cb}')
        return job_description_db


def get_question():
    questions = {"Likelihood" : """
    How well do my skills fit this job on a scale from 1-100? Again only output a single number score.
        """

    
        
    }
    return questions

def retrive_docs(resume_db, job_description_db, query):
    print(resume_db, job_description_db, query)
    if job_description_db and resume_db:
        job_description_docs = job_description_db.similarity_search(query=query,k=3)
        resume_docs = resume_db.similarity_search(query=query,k=3)
        # retrieved_docs.append(resume_docs)
        return (job_description_docs, resume_docs)

    

def main():
    prompt = ChatPromptTemplate.from_template("""
        Based on my resume data ending with a [[]]:
        Resume: {resume}
        [[]]
        and on the job description ending with a &^%:
        Job Decsription: {job_description}
        &^%
        and answer the following question:
        Question: {question}
        """)
    job_description = st.text_input("Copy and paste the upwork job description")
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    model = ChatOpenAI(model="gpt-4")
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    resume_db = get_resume(pdf)
    job_description_db = get_job_description(job_description)
    question = get_question()
    for question_name, question_text in question.items():
        try:
            job_description_docs, resume_docs = retrive_docs(resume_db, job_description_db, question_text)
        except Exception as e:
            print("this exception was caught")
            
        if job_description_docs and resume_docs and question:
            with get_openai_callback() as cb:
                response = chain.invoke({"resume": resume_docs, 
                                        "job_description": job_description_docs, 
                                        "question":question_text
                })
                print(f'response: {cb}')
            st.write(f'{question_name}: {response}')
        

if __name__ == "__main__":
    main()