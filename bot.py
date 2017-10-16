# coding: utf-8

from itertools import izip
from collections import defaultdict
import pickle
import math

import gensim
import nltk
import numpy as np
import pandas as pd

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

import common

TG_TOKEN = "4634635645:DFSSDFSDFSDFSDFSDF45654" # place token here


class IdleMain:
    def find_nearest(self, index, vect):
        max_dot = float("inf")
        max_i = None
        for i, v in enumerate(index):
            diff = v - vect
            dot = np.dot(diff, diff) / (np.linalg.norm(v) + 0.000001) / (np.linalg.norm(vect) + 0.000001)
            if dot < max_dot:
                max_dot = dot
                max_i = i
        assert max_i is not None
        return max_i, math.exp(-max_dot)


    def __init__(self):
        print "Init...",
        self.centroids = pickle.load(open("../cache/centroids.pickle"))

        data = pd.read_pickle("../cache/clusters.pickle")

        self.index = defaultdict(list)
        self.q = defaultdict(list)
        self.a = defaultdict(list)
        for label, qvect, q, a in izip(data["label"], data["qvect"], data["question"], data["answer"]):
            self.index[label].append(qvect)
            self.q[label].append(q)
            self.a[label].append(a)
        # self.qpca = pickle.load(open("../cache/qpca.pickle"))
        w2v = gensim.models.Word2Vec.load("../cache/question.w2v")
        wp = pickle.load(open("../cache/idf.pickle"))
        self.repr_calcer = common.ReprCalcer(w2v, wp)
        print "Done."


    def __call__(self, bot, update):
        try:
            s = update.message.text
            print
            print s.encode("utf-8")
            words = nltk.word_tokenize(s.lower(), language="english")
            vect = self.repr_calcer(words)
            # PCA: remove first principal component
            # vect = vect - self.qpca.inverse_transform(self.qpca.transform([vect]))[0]
            ic, c_score = self.find_nearest(self.centroids, vect)
            print len(self.index), len(self.index[ic])
            iqa, qa_score = self.find_nearest(self.index[ic], vect)
            res = "%s || %s || %s || %s || %s" % (ic, c_score, qa_score, self.q[ic][iqa], self.a[ic][iqa])
            print res
        except Exception, ex:
            print ex
            res = str(ex)
        bot.sendMessage(update.message.chat_id, text=res)


def idle_main(bot, update):
    text = update.message.text
    bot.sendMessage(update.message.chat_id, text=text)

def slash_start(bot, update):
    bot.sendMessage(update.message.chat_id, text="Hi!")

if __name__ == '__main__':
    updater = Updater(TG_TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", slash_start), group=0)
    im = IdleMain()
    dp.add_handler(MessageHandler(Filters.text, im))
    updater.start_polling()
    updater.idle()
