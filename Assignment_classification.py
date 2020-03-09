#!/usr/bin/python3

# Basic classifiction functionality with naive Bayes.
# File provided for the assignment on classification (IR course 2018/19)

# CHANGELOG
# 20171109 use python3
# 20161125 fixed bug, function evaluation sometimes fails when only a few files
# are used, because not all the categories may be represented in the test set
# 20161122 added function high_information
# s2956586 added preprocessing stopwords, stemming, punctuation, k-fold cross
# validation, high information words.


import nltk.classify
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from featx import bag_of_words, high_information_words
from classification import precision_recall
from random import shuffle
from os import listdir  # to read files
from os.path import isfile, join  # to read files
import sys
from nltk.corpus import stopwords
import string
import numpy
from nltk.probability import LaplaceProbDist
from collections import defaultdict
import itertools
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier


def get_filenames_in_folder(folder):
    """Return all the filenames in a folder."""
    return [f for f in listdir(folder) if isfile(join(folder, f))]


def read_files(categories):
    """Reads all the files that correspond to the input list
    of categories and puts their contents in bags of words."""
    feats = list()
    ps = PorterStemmer()
    print("\n##### Reading files...")
    for category in categories:
        files = get_filenames_in_folder('Volkskrant/' + category)
        num_files = 0
        for f in files:

            data = open('Volkskrant/' + category + '/' + f, 'r',
                        encoding='UTF-8').read().lower()

            translator = str.maketrans('', '', string.punctuation)

            data = data.translate(translator)

            # Filter out the dutch stopwords.
            stop_words = set(stopwords.words('dutch'))

            tokens = word_tokenize(data)

            filtered_tokens = [ps.stem(w) for w in tokens
                               if w not in stop_words]

            bag = bag_of_words(filtered_tokens)
            feats.append((bag, category))
            num_files += 1

            # you may want to de-comment this and the next line
            # if you're doing tests (it just loads N documents
            # instead of the whole collection so it runs faster

            # if num_files >= 100:
            #    break
        print ("  Category %s, %i files read" % (category, num_files))

    print("  Total, %i files read" % (len(feats)))
    return feats


def split_train_test(feats, split=0.9):
    """Splits a labelled dataset into two disjoint subsets train and test."""
    train_feats = []
    test_feats = []
    shuffle(feats)  # randomise dataset before splitting into train and test
    cutoff = int(len(feats) * split)
    train_feats, test_feats = feats[:cutoff], feats[cutoff:]
    print("\n##### Splitting datasets...")
    print("  Training set: %i" % len(train_feats))
    print("  Test set: %i" % len(test_feats))
    return train_feats, test_feats


# TODO function to split the dataset for n fold cross validation
def split_folds(feats, folds=10):
    """Splits the data in number of folds given. It then separates
    the training data from the test data."""
    shuffle(feats)  # randomise dataset before splitting into train and test
    # Divide feats into n cross fold sections
    nfold_feats = []
    train_feats = []
    test_feats = []
    train_lst = []
    for n in range(folds):
        # TODO for each fold you need 1/n of the dataset
        # as test and the rest as training
        prev_n = n / folds
        new_n = (n + 1) / folds
        train_lst.append(feats[int(len(feats) * prev_n):
                               int(len(feats) * new_n)])
    copy_train = train_lst.copy()
    for n in range(folds):
        # Seperates a test set from all sets.
        test_lst = []
        test_lst.extend(copy_train[n])
        sec_copy = copy_train.copy()
        del sec_copy[n]
        sec_copy = list(itertools.chain(*sec_copy))
        nfold_feats.append((sec_copy, test_lst))
    print("\n##### Splitting datasets...")
    return nfold_feats


def train(train_feats):
    """Trains a classifier."""
    # the following code uses the classifier with add-1 smoothing (Laplace)
    # You may choose to use that instead
    classifier = nltk.classify.NaiveBayesClassifier.train(train_feats,
                                                          estimator=LaplaceProbDist)
    return classifier


def calculate_f(precisions, recalls):
    """Calculates the f-measure of the given precisions and recall."""
    f_measures = {}
    # TODO calculate the f measure for each category
    # using as input the precisions and recalls
    precisions = change_none(precisions)
    recalls = change_none(recalls)
    for k, v in precisions.items():
        try:
            f_measures[k] = 2 * v * recalls[k] / (v + recalls[k])
        except:
            f_measures[k] = 0
    return f_measures


def change_none(dic):
    """Changes None values in dictionary to integer 0."""
    for k, v in dic.items():
        if v is None:
            dic[k] = 0
    return dic


def evaluation(classifier, test_feats, categories):
    """Prints accuracy, precision and recall, and f-measure.
    It returns the accuracy."""
    print ("\n##### Evaluation...")
    print("  Accuracy: %f" % nltk.classify.accuracy(classifier, test_feats))
    precisions, recalls = precision_recall(classifier, test_feats)
    f_measures = calculate_f(precisions, recalls)
    print(" |-----------|-----------|-----------|-----------|")
    print(" |%-11s|%-11s|%-11s|%-11s|" % ("category", "precision",
                                          "recall", "F-measure"))
    print(" |-----------|-----------|-----------|-----------|")
    for category in categories:

        if precisions[category] is None:
            print(" |%-11s|%-11s|%-11s|%-11s|" % (category, "NA", "NA", "NA"))
        else:
            print(" |%-11s|%-11f|%-11f|%-11s|" % (category,
                                                  precisions[category],
                                                  recalls[category],
                                                  round(f_measures[category],
                                                        6)))
        print(" |-----------|-----------|-----------|-----------|")

    return nltk.classify.accuracy(classifier, test_feats)


def analysis(classifier):
    """Show informative features."""
    print("\n##### Analysis...")
    # TODO show 10 most informative features
    print(classifier.show_most_informative_features(), "\n\n")


def high_information(feats, categories):
    """Obtain the high information words."""
    print("\n##### Obtaining high information words...")
    hfw_feats = []
    labelled_words = [(category, []) for category in categories]
    # 1. convert the formatting of our features to
    # that required by high_information_words
    words = defaultdict(list)
    all_words = list()
    for category in categories:
        words[category] = list()

    for feat in feats:
        category = feat[1]
        bag = feat[0]
        for w in bag.keys():
            words[category].append(w)
            all_words.append(w)
    labelled_words = [(category, words[category]) for category in categories]

    hfw_lst = list(high_information_words(labelled_words))

    # 2. calculate high information words
    for c, words in labelled_words:

        for w in words:

            if w not in hfw_lst:
                words.remove(w)
        hfw = words.copy()
        hfw = dict((el, True) for el in hfw)
        hfw_feats.append((hfw, c))

    # high_info_words contains a list of high-information words.
    # You may want to use only these for classification.
    # You can restrict the words in a bag of words to be in a
    # given 2nd list (e.g. in function read_files)
    # e.g. bag_of_words_in_set(words, high_info_words)

    print("  Number of words in the data: %i" % len(all_words))
    print("  Number of distinct words in the data: %i" % len(set(all_words)))
    print("  Number of distinct 'high-information' words in the data: %i"
          % len(hfw_feats))

    return hfw_feats


def main():
    # read categories from arguments. e.g.
    # "python3 assignment_classification.py BINNENLAND SPORT KUNST"
    categories = list()
    for arg in sys.argv[1:]:
        categories.append(arg)

    # load categories from dataset
    feats = read_files(categories)
    # hfw_feats = high_information(feats, categories)

    # TODO to use n folds you'd have to call function split_folds
    # and have the subsequent lines inside a for loop
    nfold_feats = split_folds(feats)
    accuracy = []
    MNB = SklearnClassifier(MultinomialNB())

    for train_feats, test_feats in nfold_feats:

        #  Uncomment for original classifier.
        # classifier = train(train_feats)
        # accuracy.append(evaluation(classifier, test_feats, categories))

        # If original classifier is used comment these lines.
        MNB_classifier = MNB.train(train_feats)
        accuracy.append(nltk.classify.accuracy(MNB_classifier, test_feats))

    for acc in accuracy:
        print(acc)
    print(numpy.mean(accuracy))

if __name__ == '__main__':
    main()
