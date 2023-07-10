#
# This chatbot is designed to provide relevant verses from the Noble Quran
# that may address a question, along with a potential answer derived
# from those verses using an AI-powered language model
#
# uxquran.com
#

import datetime
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings

class QuranOpenChat:

    translation = None
    qa = None

    def __init__(self, translation):
        self.translation = translation

    def initialize(self):
        current_date = datetime.datetime.now().date()
        if current_date < datetime.date(2023, 9, 2):
            llm_name = "gpt-3.5-turbo-0301"
        else:
            llm_name = "gpt-3.5-turbo"

        loader = JSONLoader(
            file_path=self.translation,
            jq_schema='.quran[]',
            content_key="text",
            metadata_func=self.metadata_func
        )
        documents = loader.load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()

        db = DocArrayInMemorySearch.from_documents(documents, embeddings)
        # db = Chroma.from_documents(docs, embeddings)

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=llm_name, temperature=0),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
        )

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["surah"] = record.get("chapter")
        metadata["aya"] = record.get("verse")
        return metadata

    def get_answer(self, question):
        if self.qa is None:
            self.initialize()

        result = self.qa({"question": question, "chat_history": []})
        source_documents = result["source_documents"]
        response = "Qn: " +  result['question']
        response = response + "\n\nReferences:\n"
        for doc in source_documents:
            content = doc.page_content
            if len(content) > 500:
                content = doc.page_content[0:500] + "..."
            metadata = doc.metadata
            sura = metadata["surah"]
            aya = metadata["aya"]
            response = response + str(sura) + ":" + str(aya) + " " + content + "\n"
            response = response + "https://uxquran.com/apps/quran-ayat/?sura=" + str(sura) + "&aya=" + str(aya) + "\n\n"

        response = response + "\nAnswer:\n"
        response = response + result['answer'] + "\n"
        return response


# chat = QuranOpenChat("/home/hashirabdulbasheer/mysite/static/input.json")
# ans=chat.get_answer("Who is God")
# print(ans)