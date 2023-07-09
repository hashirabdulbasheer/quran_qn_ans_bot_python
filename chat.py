#
# This chatbot is designed to provide relevant verses from the Noble Quran
# that may address a question, along with a potential answer derived
# from those verses using an AI-powered language model
#
# uxquran.com
#

import os
import openai
import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain


openai.api_key  = os.environ['OPENAI_API_KEY']

current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["surah"] = record.get("chapter")
    metadata["aya"] = record.get("verse")
    return metadata

def load_db(file, chain_type, k):
    # load documents
    loader = JSONLoader(
        file_path='input.json',
        jq_schema='.quran[]',
        content_key="text",
        metadata_func=metadata_func
    )
    documents = loader.load()
    # split documents
    text_splitter = CharacterTextSplitter(separator="\n", chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    );
    return qa

qa = load_db('input.json', "stuff", 10)

while True:
    try:
        print("")
        inp = input("Enter question (type 'quit' to exit):").strip()
        if inp == "quit":
            break
        result = qa({"question": inp + " from the quran context", "chat_history": []})

        source_documents = result["source_documents"]
        print("")
        print("References:")
        for doc in source_documents:
            content = doc.page_content
            if len(content) > 500:
                content = doc.page_content[0:500] + "..."
            metadata = doc.metadata
            sura = metadata["surah"]
            aya = metadata["aya"]
            print(str(sura) + ":" + str(aya) + " " + content)
            print("https://uxquran.com/apps/quran-ayat/?sura=" + str(sura) + "&aya=" + str(aya))
            print("")

        print("")
        print("Answer:")
        print(result['answer'])
    except Exception as error:
        break 

print("Thank You")
