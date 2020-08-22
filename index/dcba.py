import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents_f = open("pickled_algos/documents4.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()




word_features5k_f = open("pickled_algos/word_features5k4.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

print("loaded")
word_features.append("awesome")
def find_features(word):

    features = {}
    for w in word_features:
        features[w] = (w in word)

    return features


'''
featuresets_f = open("pickled_algos/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]
'''


open_file = open("pickled_algos/originalnaivebayes5k4.pickle", "rb")
classifier = pickle.load(open_file)
classifier.show_most_informative_features(15)
open_file.close()
print(1)

open_file = open("pickled_algos/MNB_classifier5k4.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()
print(2)



open_file = open("pickled_algos/BernoulliNB_classifier5k4.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()
print(3)


open_file = open("pickled_algos/LogisticRegression_classifier5k4.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

print(4)

open_file = open("pickled_algos/LinearSVC_classifier5k4.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()
print(5)


open_file = open("pickled_algos/SGDC_classifier5k4.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()
print(6)




voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


def feature(document):
    words = word_tokenize(document)
    pos = nltk.pos_tag(words)
    word=[]
    for a in pos:
        if(a[1][0] in ["J","R"]):
            word.append(a[0].lower())
        if(a[0] in word_features):
            word.append(a[0].lower())
    return word

def sentiment(text):

    fact=feature(text)
    feats = find_features(fact)

    if(len(fact)>0):
        return voted_classifier.classify(feats)
    else:
        return "neu"

print("input")
while(True):
    sentiment(input())