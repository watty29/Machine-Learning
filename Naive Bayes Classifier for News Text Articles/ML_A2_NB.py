#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
from collections import Counter
import time
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords

dataset = dict()                                           #putting the dataset in dic
start_time = time.clock()                                  # starting the clock time
for root, dirs, files in os.walk("20_newsgroups"):         # extracting each file from the dataset                                 
    for dir in dirs:
        dataset[dir]=[]
        for file in (os.scandir("20_newsgroups//"+dir)):
            dataset[dir].append(file.name)                 # appending each file in dataset
            
train_set =dict()           # Defining the train set
test_set =dict()            # Defining the test set

for data in dataset.keys():
    
    partition = int(len(dataset[data])*0.5)    # Divide the each dataset file into train and test set
    train_set[data] = dataset[data][0:partition]
    test_set[data] = dataset[data][partition:2*partition]

# Training the data    
words =[]
train_words =dict()    
w_count = {}
for i in train_set:
    train_words[i]=[]
    for j in train_set[i]:
        v = open("20_newsgroups//"+i +"//"+j).read().replace("\n"," ")
        for k in ",.<>/?'\";:|\}]{[+=_-)(*&^%$#@!":
            v.replace(k," ")    # replacing with ""
        train_words[i].extend(word_tokenize(v))
    w_count[i] = Counter(train_words[i])                 #  adding to the list everytime it comes across new word
    words.extend(train_words[i])  
    print(i)
words = set(words)
print(len(words))
words = list(words)

dic = {}
document_number = 1
for i in words: 
    dic[i] = {}
    for j in train_words: 
        if i in w_count[j]:
            dic[i][j] = w_count[j][i]
        else:
            dic[i][j] = 0.1
        dic[i][j]/=len(train_words[j])
        
    print("Training Class:",i)
    
probability = {}
total= 0
for i in train_set:
    probability[i] = len(train_set[i])
    total += len(train_set[i])
for j in probability:
    probability[j] = probability[j]/total


# Testing the data
stemmer = PorterStemmer()                  # reduces the word to the root/base word.
doc_probability = {}
accuracy = 0
count = 0
for data in test_set:
    for j in test_set[data]:
        
        v = open("20_newsgroups//"+data +"//"+j).read()
        test_words = word_tokenize(v)
        test_words = set(test_words)
        prob = {}
        for i in test_set:
            prob[i] = probability[i]
            for w in test_words:
                if w in dic:
                    prob[i] = prob[i]* dic[w][i]*1000
                else:
                    prob[i] = prob[i]* 0.1/len(train_words[data])*1000
        max_keys = [i for i, v in prob.items() if v == max(prob.values())]
        accuracy += (max_keys[0]==data)
        count+=1
        print("Testing Class",data)
      

print("Accuracy: ", accuracy*100/count)
print("Error: ", 100-(accuracy*100/count))   
print("Excecution time:",time.clock()-start_time)
    


# In[ ]:





# In[ ]:




