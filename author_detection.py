import io, os
import re as re
import zipfile as zipfile
import sys
import random
import math 

#Only for statistical report and data splitting
from sklearn.metrics import *

mytextzip = ''
docList=[]
idx_ID=1
author=0
with zipfile.ZipFile('30Columnists.zip') as z:
  for zipinfo in z.infolist():
    mytextzip = ''
    if zipinfo.filename.endswith('.txt') and re.search('raw_texts', zipinfo.filename):
       with z.open(zipinfo) as f:
          textfile = io.TextIOWrapper(f, encoding='cp1254', newline='')
          for line in textfile:
            if len(line.strip()): mytextzip += ' ' + line.strip()
          document = {
            'id': str(idx_ID),
            'text': mytextzip,
            'author':author
          }
          docList.append(document)
          if idx_ID % 50 == 0:
            author+=1
          idx_ID+=1
          


# TOKENIZATION

# Non-breaking to normal space
NON_BREAKING = re.compile(u"\s+"), " "
# Multiple dot
MULTIPLE_DOT = re.compile(u"\.+"), " "
# Merge multiple spaces.
ONE_SPACE = re.compile(r' {2,}'), ' '
# Numbers
NUMBERS= re.compile(r'[0-9]*[0-9]'), ' ' 
# 2.5 -> 2.5 - asd. -> asd . 
DOT_WITHOUT_FLOAT = re.compile("((?<![0-9])[\.])"), r' '
# 2,5 -> 2,5 - asd, -> asd , 
COMMA_WITHOUT_FLOAT = re.compile("((?<![0-9])[,])"), r' '
# doesn't -> doesn't  -  'Something' -> ' Something '
QUOTE_FOR_NOT_S = re.compile("[\']"), r' '
AFTER_QUOTE_SINGLE_S = re.compile("\s+[s]\s+"), r' '
# Extra punctuations "!()
NORMALIZE = re.compile("([\–])"), r'-'
EXTRAS_PUNK = re.compile("([^\'\.\,\w\s\-\–])"), r' '

STOP_WORDS_LIST=['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
STOP_WORDS=re.compile(r'\b(?:%s)\b[^\-]' % '|'.join(STOP_WORDS_LIST)),r" "

REGEXES = [
    STOP_WORDS,
    NON_BREAKING,
    MULTIPLE_DOT,
    NUMBERS,
    DOT_WITHOUT_FLOAT,
    COMMA_WITHOUT_FLOAT,
    QUOTE_FOR_NOT_S,
    AFTER_QUOTE_SINGLE_S,
    NORMALIZE,
    EXTRAS_PUNK,
    ONE_SPACE
]

def pre_porcess_tokenize_sentence(sentence):
  sentence = sentence.lower()
  for regexp, subsitution in REGEXES:
    sentence = regexp.sub(subsitution, sentence)   
  return sentence

import time
start_time = time.time()

tokenizedList=[]
for doc in docList:
  tokenizedText = pre_porcess_tokenize_sentence(doc['text'])
  tokens = tokenizedText.split(' ')
  del tokens[0]
  del tokens[len(tokens)-1]
  tokenizedList.append(tokens)

elapsed_time = time.time() - start_time

print("Tokenize: "+str(elapsed_time))
#print(tokenizedList)

# DOCUMENT VECTOR
start_time = time.time()
wordsFreqMatrix={}

docIdx=0
for wordLists in tokenizedList:
  for word in wordLists:
    if word in wordsFreqMatrix.keys():
      wordsFreqMatrix[word][docIdx] += 1
    else:
      wordsFreqMatrix[word]=[0 for i in range(0,1500)]
      wordsFreqMatrix[word][docIdx] += 1
  docIdx += 1

doc2vec={i:[row[i] for row in wordsFreqMatrix.values()] for i in range(docIdx)}

elapsed_time = time.time() - start_time
print("Document Vector: ",str(elapsed_time))


# Cosine similarity
def CossineSimilarity(x, y): 
  multiplication=[a*b for a,b in zip(x,y)]
  totalTop=sum(multiplication)
  squareA=sum([a*a for a in x])
  squareB=sum([a*a for a in y])
  rootSumm=math.sqrt(squareA)+math.sqrt(squareB)
  distance=totalTop/rootSumm
  return math.sqrt(distance) 

#KNN Algorithm
def classifyDict(trainX,trainY,testX,k=10,clsCount=3):
    distances={}
    for i in range(0,len(trainX)):
        distances[i]=CossineSimilarity(trainX[i],testX)
    sortedDistance= {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    sortedDistanceListForK=list(sortedDistance.keys())[:k]
    freqs=[0 for i in range(clsCount)]
    for i in range(len(sortedDistanceListForK)):
      freqs[trainY[sortedDistanceListForK[i]]]+=1
    maxC=max(freqs)
    return freqs.index(maxC)

def test_train_split(wholeData,doc2vecMatrix,testCount,classCount,classLabel):
  test={}
  reach=testCount*classCount
  created=[0 for item in range(classCount)]
  while sum(created) < reach:
    randomID = random.randint(0,len(wholeData)-1)
    if randomID in test.keys():
      continue
    else:
      if created[wholeData[randomID][classLabel]]<testCount:
        test[randomID+1]=wholeData[randomID]
        created[wholeData[randomID][classLabel]]+=1
  testX=[doc2vecMatrix[item-1] for item in test.keys()]
  testY=[item[classLabel] for item in test.values()]
  trainX=[doc2vecMatrix[idx] for idx in range(len(doc2vecMatrix)) if idx+1 not in test.keys()]
  trainY=[item[classLabel] for item in wholeData if int(item['id']) not in test.keys()]
  return trainX, testX, trainY, testY

trainX, testX,trainY, testY = test_train_split(docList,doc2vec,2,30,'author')
resultsArray=[]
for j in range(0,len(testX)):
  start_time = time.time()
  resultsArray.append(classifyDict(trainX,trainY,testX[j],clsCount=30))
  elapsed_time = time.time() - start_time
  print("Classifier %d :" % (j),str(elapsed_time))


conf_matrix=confusion_matrix(testY, resultsArray)
classes = [i for i in range(1,31)]
print("\t",end='')
for label in classes:
    print("{:<4}".format(label),end='')
label=0
print()
for idx in conf_matrix:
    print("{:<4}".format(classes[label]),end='')
    for i in range(len(classes)):
        print("{:<4}".format(idx[i]),end='')
    label+=1
    print()

accurarcy=accuracy_score(testY, resultsArray)
print(accurarcy)

