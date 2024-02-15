from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
import streamlit as st
from pypdf import PdfReader


"""
1. Please enter your OpenAI API Key to use the app
"""

questions = {"Fitness Score" : """
    How well do my skills fit this job on a scale from 1-100? Only output a single number score. If you cannot calculate a score put N/A.
        """,
    "Estimated Salary" : """
    Give an estimated salary for this job based on the job description and explain the resposnibilities that it might ential. 
    """ ,
    "Cover Letter" : """
    Use the information from the resume to create a well written cover letter for the described job position.
""",
    "Gaps" : """
    Use the informaiton in the resume to create a list of gaps in the users experience
""",
    "Day-to-day Activities" : """
    Use the information in the job description to create a list of day-to-day activities that the user might have to do in the described job position.
"""
    }



job_description_example = '''
The project is to analyze the nonces submitted by different types of hardware and create an identifying signature for each of the hardware.
An expert ML person is required.
Person should be an expert programmer and be able to pull data, do the anlaysis, build the models and create a usable dashboard or excel for the end-user.
Good knowledge of blockchain and proof-of-work mining will be desirable.
Good knowledge of c++ or firmware is desirable.
Person should be available full-time and be flexible for calls over weekends and late evenings
'''


resume_db = None
job_description = None
resume_retriever = None

open_ai_api_key = st.text_input("Enter your OpenAI API Key to use")


if open_ai_api_key != "":
    embeddings = OpenAIEmbeddings(api_key=open_ai_api_key)
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4", api_key=open_ai_api_key, temperature=0)


if open_ai_api_key != "":
    st.write("2. You can now upload your resume")
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None and open_ai_api_key != "":
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
        resume_db = FAISS.from_documents(docs, embeddings)
        st.write("Resume uploaded")
        resume_retriever = resume_db.as_retriever()

if open_ai_api_key != "" and resume_db is not None:
    st.write('3. Now you can copy and paste the job description')
    job_description = st.text_input("Copy and paste the upwork job description")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if resume_retriever is not None and job_description != "" and job_description is not None and open_ai_api_key is not None:
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=resume_retriever, return_source_documents=False)
    st.write("Calculating the score")
    for category, question in questions.items():
        result = qa.run({'query': question, 
                        'context':job_description
                        })
        st.write(category, ":", result)
  


