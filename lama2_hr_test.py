# -*- coding: utf-8 -*-
"""lama2-hr-test.ipynb


"""

!pip install gradio
!pip install python_dotenv
!pip install langchain
!pip install pinecone-client
!pip install sentence_transformers
!pip install pdf2image
!pip install pypdf
!pip install xformers
!pip install bitsandbytes accelerate transformers
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import os
import sys
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA

from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA

!mkdir pdf-area

loader=PyPDFDirectoryLoader('/content/pdf-area')
data=loader.load()

print(data[0])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

docs=text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

query_result = embeddings.embed_query("Hello World",)
print("Length", len(query_result))

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', '')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = "lama2" # put in the name of your pinecone index here

docsearch=Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)

docsearch = Pinecone.from_existing_index(index_name, embeddings)

query="summarize the leave policy"
docs=docsearch.similarity_search(query,k=4)

docs

from huggingface_hub import notebook_login
import torch

notebook_login()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",use_auth_token=True)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                             load_in_8bit=True
                                             )

pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0.1,},)

SYSTEM_PROMPT = """Use the following pieces of context to answer the question at the end.
Act as AI language model, if you dont know the answer, simple say you dont."""
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<>\n", "\n<>\n\n"

SYSTEM_PROMPT = B_SYS + SYSTEM_PROMPT + E_SYS
instruction = """
{context}

Question: {question}
"""

template = B_INST + SYSTEM_PROMPT + instruction + E_INST
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

result = qa_chain("tell me leave policy")

result['result']

import sys
import gradio as gr
import time

def answer_question(user_input):
    if user_input == 'exit':
        print('Exiting')
        sys.exit()
    if user_input == '':
        return "Please enter a valid question."

    result = qa_chain({'query': user_input})
    response_text = result['result']

    # Simulate typing effect
    typed_response = ""
    for char in response_text:
        typed_response += char
        time.sleep(0.05)

        print("\r" + typed_response, end="")
        sys.stdout.flush()

    print()

    return response_text

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question", type="text", lines=10),
    outputs=gr.Textbox(label="Answer", type="text", lines=10),
    title="Interloop HR BOT  ",
    description="Ask a question and get an answer from the chatbot",
    theme="dark"
)

iface.launch()

