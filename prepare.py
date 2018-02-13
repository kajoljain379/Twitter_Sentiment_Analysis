#from replaceExpand import *
from collections import defaultdict

def loadDictionary():
    """create emoticons dictionary"""
    f=open("emoticonsWithPolarity.txt",'r')
    data=f.read().split('\n')
    emoticonsDict={}
    for i in data:
        if i:
            i=i.split()
            value=i[-1]
            key=i[:-1]
            for j in key:
                emoticonsDict[j]=value
    f.close()
    #storedlike = {'0v0': 'Extremely-Positive', ':/': 'Negative', ':(': 'Negative', ':)': 'Positive', ':*': 'Positive', '(._.)': 'Negative', ':-@[1]': 'Negative'}

    #print emoticonsDict
    specialChar='1234567890#@%^&()_=`{}:"|[]\;\',./\n\t\r '
    """create acronym dictionary"""
    f=open("acronym_tokenised.txt",'r')
    data=f.read().split('\n')
    acronymDict={}
    for i in data:
        if i:
            i=i.split('\t')
            word=i[0].split()
            token=i[1].split()[1:]
            key=word[0].lower().strip(specialChar)
            value=[j.lower().strip(specialChar) for j in word[1:]]
            acronymDict[key]=[value,token]
    f.close()
    #storedlike = 'ukwim': [['you', 'know', 'what', 'i', 'mean'], ['O', 'V', 'O', 'O', 'V']]
    #print acronymDict

    """create stopWords dictionary"""
    stopWords=defaultdict(int)
    f=open("stopWords.txt", "r")
    for line in f:
        if line:
            line=line.strip(specialChar).lower()
            stopWords[line]=1
    f.close()
    #storedlike = {'all': 1, "she'll": 1, 'being': 1}

    return acronymDict,stopWords,emoticonsDict