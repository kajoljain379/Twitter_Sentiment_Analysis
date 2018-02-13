import sys
import nltk
import csv
import re
import nltk.metrics
import itertools
import collections
import nltk.classify.util
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from prepare import *
from collections import defaultdict
from nltk.stem import PorterStemmer
ps = PorterStemmer()

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
#     bigram_finder = BigramCollocationFinder.from_words(words)
#     bigrams = bigram_finder.nbest(score_fn, n)
#     return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

# def replaceall(words):
#     newline=[]
#     for word in words:
#         if word in POSITIVE:
#             newline.append("good")
#             continue
#         if word in NEGATIVE:
#             newline.append("bad")
#             continue
#         newline.append(word.lower())
#     return newline


def main():
    pos_tweets=[]
    neg_tweets=[]
    pos_test=[]
    neg_test=[]
    specialChar='1234567890#@%^&()_=`{}:"|[]\;\',./\n\t\r '
    acronymDict,stopWords,emoticonsDict = loadDictionary()

    # POSITIVE = ["*O", "*-*", "*O*", "*o*", "* *",
    #             ":P", ":D", ":d", ":p",
    #             ";P", ";D", ";d", ";p",
    #             ":-)", ";-)", ":=)", ";=)",
    #             ":<)", ":>)", ";>)", ";=)",
    #             "=}", ":)", "(:;)",
    #             "(;", ":}", "{:", ";}",
    #             "{;:]",
    #             "[;", ":')", ";')", ":-3",
    #             "{;", ":]",
    #             ";-3", ":-x", ";-x", ":-X",
    #             ";-X", ":-}", ";-=}", ":-]",
    #             ";-]", ":-.)",
    #             "^_^", "^-^"]
    # NEGATIVE = [":(", ";(", ":'(",
    #             "=(", "={", "):", ");",
    #             ")':", ")';", ")=", "}=",
    #             ";-{{", ";-{", ":-{{", ":-{",
    #             ":-(", ";-(",
    #             ":,)", ":'{",
    #             "[:", ";]"
    #             ]
    # posemoji = '|'.join(POSITIVE)
    # negemoji = '|'.join(NEGATIVE)
    
    #print(acronymDict)
    #print(stopWords)
    #print(emoticonsDict)
    with open('newsent.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            if line['sentiment']=="Positive":
                #newline=line['text']
                #newline = re.sub(posemoji, 'good', newline)
                #newline = re.sub(negemoji, 'bad', newline)
                words=re.sub('[^A-Za-z0-9]+', ' ', line['text'])
                #words=line['text']
                pos_tweets.append(words)
            if line['sentiment']=="Negative":
                #newline=line['text']
                #newline = re.sub(posemoji, 'good', newline)
                #newline = re.sub(negemoji, 'bad', newline)
                words=re.sub('[^A-Za-z0-9]+', ' ', line['text'])
                #words=line['text']
                neg_tweets.append(words)
    #f=open(sys.argv[1],'r')
    #f.close()
    with open('newtest.csv') as tcsvfile:
        reader = csv.DictReader(tcsvfile)
        for line in reader:
            if line['sentiment']=="Positive":
                #newline=line['text']
                #newline = re.sub(posemoji, 'good', newline)
                #newline = re.sub(negemoji, 'bad', newline)
                words=re.sub('[^A-Za-z0-9]+', ' ', line['text'])
                #words=line['text']
                pos_test.append(words)
            if line['sentiment']=="Negative":
                #newline=line['text']
                #newline = re.sub(posemoji, 'good', newline)
                #newline = re.sub(negemoji, 'bad', newline)
                words=re.sub('[^A-Za-z0-9]+', ' ', line['text'])
                #words=line['text']
                neg_test.append(words)
    #print(pos_tweets[:6])
    

    # #Emoticons handling
    # tok_postweets=[]
    # tok_postest=[]
    # tok_negtweets=[]
    # tok_negtest=[]
    # for (words) in pos_tweets:
    #     words_filtered=replaceall(words)
    #     tok_postweets.append((words_filtered))
    # for (words) in neg_tweets:
    #     words_filtered=replaceall(words)
    #     tok_negtweets.append((words_filtered))
    # for (words) in pos_test:
    #     words_filtered=replaceall(words)
    #     tok_postest.append((words_filtered))
    # for (words) in neg_test:
    #     words_filtered=replaceall(words)
    #     tok_negtest.append((words_filtered)) 

    #Tokenizing
    tok_postweets=[]
    tok_postest=[]
    tok_negtweets=[]
    tok_negtest=[]
    for (words) in pos_tweets:
        words_filtered = [e.lower() for e in words.split()]
        tok_postweets.append((words_filtered))
    for (words) in neg_tweets:
        words_filtered = [e.lower() for e in words.split()]
        tok_negtweets.append((words_filtered))
    for (words) in pos_test:
        words_filtered = [e.lower() for e in words.split()]
        tok_postest.append((words_filtered))
    for (words) in neg_test:
        words_filtered = [e.lower() for e in words.split()]
        tok_negtest.append((words_filtered)) 
    #print(tok_postweets[:1])
    #print(tok_negtweets[:1])
    
        #Stopwords Removal
    

    filter_tweetspos=[]
    filter_tweetsneg=[]
    filter_testpos=[]
    filter_testneg=[]
    stopWords=defaultdict(int)
    stopword=[]
    f=open("stopWords.txt", "r")
    for line in f:
        if line:
            line=line.strip(specialChar).lower()
            stopword.append(line)
    f.close()
    #print(stopword)
    for sent in tok_postweets:
        filtered_sentence=[]
        for w in sent:
            #print(w)
            if w not in stopword:
                #print(w)
                if "http" not in w:
                    filtered_sentence.append(w)
        filter_tweetspos.append(filtered_sentence)
    for sent in tok_negtweets:
        filtered_sentence=[]
        for w in sent:
            #print(w)
            if w not in stopword:
                #print(w)
                if "http" not in w:
                    filtered_sentence.append(w)
        filter_tweetsneg.append(filtered_sentence)
    for sent in tok_postest:
        filtered_sentence=[]
        for w in sent:
            #print(w)
            if w not in stopword:
                if "http" not in w:
                #print(w)
                    filtered_sentence.append(w)
        filter_testpos.append(filtered_sentence)
    for sent in tok_negtest:
        filtered_sentence=[]
        for w in sent:
            #print(w)
            if w not in stopword:
                if "http" not in w:
                #print(w)
                    filtered_sentence.append(w)
        filter_testneg.append(filtered_sentence) 
    #print(filter_tweetspos[:1])

     #Stemming
    pos_tweets=[]
    neg_tweets=[]
    pos_test=[]
    neg_test=[]
    for sent in filter_tweetspos:
        filtered_sentence=[]
        for w in sent:
            filtered_sentence.append(ps.stem(w))
        pos_tweets.append(filtered_sentence)
    for sent in filter_tweetsneg:
        filtered_sentence=[]
        for w in sent:
            filtered_sentence.append(ps.stem(w))
        neg_tweets.append(filtered_sentence)
    for sent in filter_testpos:
        filtered_sentence=[]
        for w in sent:
            filtered_sentence.append(ps.stem(w))
        pos_test.append(filtered_sentence)
    for sent in filter_testneg:
        filtered_sentence=[]
        for w in sent:
            filtered_sentence.append(ps.stem(w))
        neg_test.append(filtered_sentence) 
    #print(pos_tweets)


    # #Bigram
    # bitweets=[]
    # bitest=[]
    # for line in pos_tweets:
    #     #line=line.lower()
    #     #line=removePunctuation(line)
    #     newline = [e.lower() for e in line]
    #     #print(newline)
    #     line=' '.join(newline)
    #     #print(line)
    #     words_filtered = [' '.join(item) for item in nltk.bigrams (line.split())]
    #     bitweets.append((words_filtered,'positive'))
    # for line in neg_tweets:
    #     #line=line.lower()
    #     #line=removePunctuation(line)
    #     newline = [e.lower() for e in line]
    #     line=' '.join(newline)
    #     words_filtered = [' '.join(item) for item in nltk.bigrams (line.split())]
    #     bitweets.append((words_filtered,'negative'))
    # for line in pos_test:
    #     #line=line.lower()
    #     #line=removePunctuation(line)
    #     newline = [e.lower() for e in line]
    #     line=' '.join(newline)
    #     words_filtered = [' '.join(item) for item in nltk.bigrams (line.split())]
    #     bitest.append((words_filtered,'positive'))
    # for line in neg_test:
    #     #line=line.lower()
    #     #line=removePunctuation(line)
    #     newline = [e.lower() for e in line]
    #     line=' '.join(newline)
    #     words_filtered = [' '.join(item) for item in nltk.bigrams (line.split())]
    #     bitest.append((words_filtered,'negative'))
    # biword_features = get_word_features(get_words_in_tweets(bitweets))
    # bitest_features = get_word_features(get_words_in_tweets(bitest))

    tweets=[]
    test=[]

    for (words) in pos_tweets:
        #print(words)
        words= [e.lower() for e in words]
        tweets.append((words,'positive'))
    for (words) in neg_tweets:
        words= [e.lower() for e in words]
        tweets.append((words,'negative'))
    for (words) in pos_test:
        words= [e.lower() for e in words]
        test.append((words,'positive'))
    for (words) in neg_test:
        words= [e.lower() for e in words]
        test.append((words,'negative'))
    #print(tweets)

    word_features = get_word_features(get_words_in_tweets(tweets))
    test_features = get_word_features(get_words_in_tweets(test))

    def extract_features(document):
        document_words = set(document)
        features={}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    def extract_features1(document):
        document_words = set(document)
        features={}
        for word in test_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    training_set=nltk.classify.apply_features(extract_features,tweets)
    testing_set=nltk.classify.apply_features(extract_features1,test)


    #evaluate_classifier(bigram_word_feats)

    #NB
    classifier=nltk.NaiveBayesClassifier.train(training_set)
    print('accuracy:',(nltk.classify.util.accuracy(classifier,testing_set))*100)
    #result = 0.8244563504569807
    #result = 0.8247715096123542
        
    #BNB
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
    #result = 81.44796380090497

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
if __name__ == "__main__":                                                                              
    main()
