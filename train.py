import math
import os
import pickle

import gensim
import nltk

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

import common


if __name__ == "__main__":
    try:
        os.mkdir("../cache")
    except OSError as err:
        print err

    print "load..."
    data = pd.read_csv(
        "../data/vk.csv",
        dtype = {
            "question": str,
            "answer": str,
        },
    )

    print "tokenize..."
    data["question_tokens"] = data["question"].map(lambda s: nltk.word_tokenize(str(s).decode("utf-8").lower(), language="english"))
    data["question_tokens_str"] = data["question_tokens"].map(lambda s: "_".join(s).encode("utf-8"))

    data["answer_tokens"] = data["answer"].map(lambda s: nltk.word_tokenize(str(s).decode("utf-8").lower(), language="english"))
    data["answer_tokens_str"] = data["answer_tokens"].map(lambda s: "_".join(s).encode("utf-8"))

    if True:
        count = common.Count()
        data["question_tokens"].map(count)
        data["answer_tokens"].map(count)
        WP = count.finalize(2*len(data))
        pickle.dump(WP, open("../cache/idf.pickle", "wb"))
    else:
        WP = pickle.load(open("../cache/idf.pickle"))


    if True:
        word2vec = gensim.models.Word2Vec(
            pd.concat([data["question_tokens"], data["question_tokens"], data["answer_tokens"]]),
            size=100, window=5, min_count=10, workers=16
        )
        word2vec.save("../cache/question.w2v")
    else:
        word2vec = gensim.word2vecs.Word2Vec.load("../cache/question.w2v")

    print "qvect..."

    repr_calcer = common.ReprCalcer(word2vec, WP)

    data["qvect"] = data["question_tokens"].map(repr_calcer)
    data["avect"] = data["answer_tokens"].map(repr_calcer)

    # PCA: remove first principal component
    # qX = np.array(data["qvect"].tolist())
    # qpca = PCA(n_components=1)
    # qpca.fit(qX)
    # qX = qX - qpca.inverse_transform(qpca.transform(qX))
    # data["qvect"] = [x for x in qX]
    # pickle.dump(qpca, open("../cache/qpca.pickle", "wb"))

    print "k-means..."
    X = np.array(list(data["qvect"]))
    print X.shape
    kmeans = KMeans(n_clusters=1000, random_state=0, n_jobs=16).fit(X)
    print kmeans.labels_
    data["label"] = kmeans.labels_

    data.sort_values("label", inplace=True)

    print "save..."
    # data.to_csv("../cache/clusters.csv", columns=["label", "question", "answer", "question_tokens_str", "qvect"], index=False)
    data.to_pickle("../cache/clusters.pickle")

    # np.savetxt("../cache/centroids.csv", kmeans.cluster_centers_, delimiter=",")
    pickle.dump(kmeans.cluster_centers_, open("../cache/centroids.pickle", "wb"))
