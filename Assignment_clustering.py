#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic clustering functionality with K-Means.
File provided for the assignment on clustering (IR course 2018/19)
"""

from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
from os import listdir  # to read files
from os.path import isfile, join  # to read files
import sys
import itertools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re


def rss_plot():
    """Creates a plot with residuals of sum of squared distances(RSS) where
    the y-axis is RSS and the x-axis is number of clusters."""
    print("###### Creating RSS plot")
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # All RSS calculations given trough model.inertia_ in order with clusters.
    RSS = [2788.2119991615964, 2721.689727564284, 2686.8343093013154,
           2662.5505307606286, 2643.7207663851864, 2627.5351713628015,
           2613.0617146180307, 2601.1243761445626, 2588.323251601801,
           2575.6477117186673, 2562.371784135395, 2555.382899858694,
           2544.606693745832, 2534.869786211854, 2522.14783183395]
    m = [k, RSS]

    # Creates plot
    plt.title("RSS of K-means algorithm ")
    plt.plot(m[0], m[1], label="RSS")
    plt.xlabel("Number of clusters. (K-Means)")
    plt.ylabel("RSS")
    plt.legend()
    plt.show()


def show_plot():
    """Shows a plot with purity and rand-index scores on the y-axis. On
    the x-axis the number of clusters is shown. It also adds a title and
    legenda."""
    p = [0.802, 0.787, 0.813, 0.833]
    r = [0.8196232419570345, 0.7177300564937577, 0.6762154995430005,
         0.6417119858702327, 2544.606693745832]
    k = [3, 5, 10, 20]
    m = [k, p, r]

    # Creates plot
    plt.title("Cluster, purity,and rand-index relationships.")
    plt.plot(m[0], m[1], label="Purity")
    plt.plot(m[0], m[2], label="Rand-index")
    plt.xlabel("Number of clusters. (K-Means)")
    plt.ylabel("Purity/Rand-index")
    plt.legend()
    plt.show()


def get_filenames_in_folder(folder):
    """Return all the filenames in a folder."""
    return [f for f in listdir(folder) if isfile(join(folder, f))]


def read_files(categories):
    """Reads all the files that correspond to the input list
    of categories. The following pre-processing is done:
    'lowerd, stripped from whitespace, removed punctuation,
    removed stop words, tokenised and stemmed the text with
    the Snowball Stemmer for Dutch words'."""
    print("\n##### Reading files...")
    stemmer = SnowballStemmer("dutch")
    stop_words = set(stopwords.words('dutch'))
    X = list()
    y = list()
    for category in categories:
        files = get_filenames_in_folder('Volkskrant/' + category)
        num_files = 0
        for f in files:
            data = open('Volkskrant/' + category + '/' + f, 'r',
                        encoding='UTF-8').read().lower().strip()
            translator = str.maketrans('', '', string.punctuation)
            data = data.translate(translator)
            tokens = word_tokenize(data)
            filtered_tokens = [stemmer.stem(w) for w in tokens
                               if w not in stop_words]
            sentence = " ".join(filtered_tokens)
            sentence = re.sub(r'\d+', '', sentence)
            X.append(sentence)
            y.append(category)
            num_files += 1
            # if num_files>=50: # you may want to de-comment this and the
            # next line if you're doing tests (it just loads N documents
            # instead of the whole collection so it runs faster
            # break
        print ("  Category %s, %i files read" % (category, num_files))
    print("  Total, %i files read" % (len(X)))
    return X, y


def prepare_data(X, n_features=1000):
    """Transforms the data into vectors."""
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize,
                                 max_features=n_features)
    X_prep = vectorizer.fit_transform(X)
    return X_prep, vectorizer


def kmeans(X_prep, n_k=3):
    """Returns a clustering model and the number of clusters."""
    print("\n##### Running K-Means...")
    km = KMeans(init="random", n_clusters=n_k, verbose=False,
                precompute_distances=True, n_jobs=-1)
    km.fit_predict(X_prep)
    return km, n_k


def evaluate(model, y):
    """Prints a contingency matrix, purity score, adjusted
    rand-index and the rand-index."""
    print("\n##### Evaluating...")
    print("Contingency matrix")
    contingency_matrix = metrics.cluster.contingency_matrix(y,
                                                            model.labels_)
    print (contingency_matrix)
    purity = np.sum(np.amax(contingency_matrix,
                            axis=0)) / np.sum(contingency_matrix)
    print("Purity %.3f" % purity)
    print("Adjusted rand-index: %.3f"
          % metrics.adjusted_rand_score(y, model.labels_))

    # TODO calculate the rand index
    # Use the cluster IDs (model.labels_) and the categories (y)
    print("\n##### Calculating rand index...")
    labels_outcome = []
    y_outcome = []
    for a, b in itertools.combinations(model.labels_, 2):
        if a != b:
            labels_outcome.append(False)
        else:
            labels_outcome.append(True)
    for a, b in itertools.combinations(y, 2):
        if a != b:
            y_outcome.append(False)
        else:
            y_outcome.append(True)
    print("Rand-index:\t", rand_index(labels_outcome, y_outcome), "\n")


def rand_index(l1, l2):
    """Calculates and returns the rand index when given two lists which contain
    True and False statements."""
    it1 = iter(l1)
    it2 = iter(l2)
    n1 = next(it1)
    n2 = next(it2)
    outcomes = []
    while True:
        try:
            if n1 is False and n2 is False:
                outcomes.append("TN")
                n1 = next(it1)
                n2 = next(it2)
            elif n1 is False and n2 is True:
                outcomes.append("FP")
                n1 = next(it1)
                n2 = next(it2)
            elif n1 is True and n2 is False:
                outcomes.append("FN")
                n1 = next(it1)
                n2 = next(it2)
            else:
                outcomes.append("TP")
                n1 = next(it1)
                n2 = next(it2)
        except StopIteration:
            tp = outcomes.count("TP")
            tn = outcomes.count("TN")
            fp = outcomes.count("FP")
            fn = outcomes.count("FN")
            ri = (tp+tn) / (tp + fp + fn + tn)
            return ri


# TODO complete this function
def top_terms_per_cluster(model, vectorizer, n_k, n_terms=10):
    """Prints top 10 terms per cluster."""
    print("\n##### Top terms per cluster...")
    sorted_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    # TODO iterate over each cluster
    # then get the ids of the 10 top terms in the cluster from sorted_centroids
    # then get the terms that correspond to the term ids (from terms)
    for i in range(n_k):
        print("Cluster %d:" % (i+1)),
        for ind in sorted_centroids[i, :n_terms]:
            print(' %s' % terms[ind])
        print("\n")


# TODO complete this function
def hierarchical_clustering(X_prep, n_k=3):
    """Peforms a hierachical clustering algorithm and
    returns the model."""
    print("\n##### Running hierarchical clustering...")
    X_prep = X_prep.toarray()

    # TODO run hierarchical clustering and return the resulting model
    hc = AgglomerativeClustering(n_clusters=n_k, linkage='average')
    hc.fit_predict(X_prep)
    return hc


def main():
    # show_plot()
    categories = list()
    for arg in sys.argv[1:]:
        categories.append(arg)
    X, y = read_files(categories)
    X_prep, vectorizer = prepare_data(X)
    model, n_k = kmeans(X_prep)

    # TODO call rand_index and top terms
    top_terms_per_cluster(model, vectorizer, n_k, n_terms=10)
    evaluate(model, y)

    # TODO call evaluate and rand_index
    hc_model = hierarchical_clustering(X_prep)
    evaluate(hc_model, y)

    # Show plot of residuals of sum of squared residuals.
    rss_plot()
if __name__ == '__main__':
    main()
