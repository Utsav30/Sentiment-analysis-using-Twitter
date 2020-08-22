from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.db import models

# Create your views here.
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
#**************************************************************************************************************8

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
from pathlib import Path



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


documents_f = open( Path(__file__).parent /"pickled_algos/documents4.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()




word_features5k_f = open( Path(__file__).parent /"pickled_algos/word_features5k4.pickle", "rb")
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


open_file = open( Path(__file__).parent /"pickled_algos/originalnaivebayes5k4.pickle", "rb")
classifier = pickle.load(open_file)
classifier.show_most_informative_features(15)
open_file.close()
print(1)

open_file = open(Path(__file__).parent /"pickled_algos/MNB_classifier5k4.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()
print(2)



open_file = open(Path(__file__).parent /"pickled_algos/BernoulliNB_classifier5k4.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()
print(3)


open_file = open(Path(__file__).parent /"pickled_algos/LogisticRegression_classifier5k4.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

print(4)

open_file = open(Path(__file__).parent /"pickled_algos/LinearSVC_classifier5k4.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()
print(5)


open_file = open(Path(__file__).parent /"pickled_algos/SGDC_classifier5k4.pickle", "rb")
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




# ************************************************************************************************************
class TwitterClient(object):


    def __init__(self):

        # keys and tokens from the Twitter Dev Console
        consumer_key = '1WbF21ELogv2QC71aeB5vAyYC'
        consumer_secret = 'kuxTkXemsGIgpLbQcz62CNTUqmlpIeS2tNk4QkDynD1NuC4BQ5'
        access_token = '897780326302273536-pNIgDH7CDDdZiZPn88ieMEaoh9D01Pg'
        access_token_secret = '2Xo66tbN5FRMflcl5FzLs48YNPMDxsUnyPafFo3yv54Jf'

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):

        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):

        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity >0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, count=10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []

        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q=query, count=count)

            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}

                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

                    # return parsed tweets
            return tweets

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))



def search(topic):
    api = TwitterClient()
    # calling function to get tweets
    tweets = api.get_tweets(query=topic, count=500)

    # picking positive tweets from tweets
    if tweets!= None:
        ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
        # percentage of positive tweets
        print("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))

        # picking negative tweets from tweets
        ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
        # percentage of negative tweets
        print("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))
        # percentage of neutral tweets
        print("Neutral tweets percentage: {} % ".format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets)))

        # printing first 5 positive tweets
        print("\n\nPositive tweets:")
        pt=[]
        nt=[]
        for tweet in ptweets[:10]:
            print(tweet['text'])
            pt.append(tweet['text'])

            # printing first 5 negative tweets
        print("\n\nNegative tweets:")
        for tweet in ntweets[:10]:
            print(tweet['text'])
            nt.append(tweet['text'])

        return (100 * len(ptweets) / len(tweets)), (100 * len(ntweets) / len(tweets)), 100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets),pt[:5],nt[:5],len(tweets), len(ntweets), len(ptweets)

    print("Error")
    
def index(request):
   # tamplat = loader.get_template('index/index.html')


    if request.method=="POST":
        topic=request.POST.get('text')
        print(topic)
        sent=search(topic)
        if sent!=None:
            positive,negative,neutral,pt1,nt1,tt,nt,pt=sent[0],sent[1],sent[2],sent[3],sent[4],sent[5],sent[6],sent[7]# search(topic)
            nnt=tt-nt-pt
            return render(request, "./index/index.html" , {"topic":topic.upper(), "neg": negative, "pos": positive, "nu":neutral, "pt0":pt1, "nt0":nt1, "tt":tt, "nnt":nnt, "nt":nt, "pt":pt})


    return render(request,"./index/index1.html",{})
