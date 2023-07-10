#
# This chatbot is designed to provide relevant verses from the Noble Quran
# that may address a question, along with a potential answer derived
# from those verses using an AI-powered language model
#
# uxquran.com
#

import numpy as np
import json
import itertools
from langchain.embeddings.openai import OpenAIEmbeddings

class QuranSimilarVerses:
    quran_data = None
    translation = None
    embedding = OpenAIEmbeddings()

    def __init__(self, quran_embeddings, translation):
        self.quran_data = np.load(quran_embeddings, allow_pickle=True)
        trans_file = open(translation)
        trans_json = json.load(trans_file)
        trans_file.close()
        self.translation = trans_json


    def get_verse(self, surah, aya):
        return [d for d in self.translation["quran"] if int(d['chapter']) == int(surah) and int(d['verse']) == int(aya)][0]["text"]


    def get_similar(self, question):
        if self.quran_data is None:
            return "Error: Loading data, please try again after some time"

        embedding_input_string = self.embedding.embed_query(question)
        result = {}
        for key in self.quran_data.item():
            result[key] = np.dot(embedding_input_string, self.quran_data.item()[key])

        # sorting in descending order
        sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

        response = "Qn: " +  question
        response = response + "\nVerses:\n"

        # get top N results
        N = 4
        sliced_result = dict(itertools.islice(sorted_result.items(), N))

        for item in sliced_result:
            indices = item.split("_")
            surah = indices[0]
            aya = indices[1]
            response = response + str(surah) + ":" + str(aya) + " " + self.get_verse(surah, aya) + "\n"
            response = response + "https://uxquran.com/apps/quran-ayat/?sura=" + surah + "&aya=" + aya + "\n\n"

        return response


quran = QuranSimilarVerses("out_wahiddudin.npy", "input.json")
answer = quran.get_similar("How many years did Ashabul Khaf sleep in the cave")
print(answer)