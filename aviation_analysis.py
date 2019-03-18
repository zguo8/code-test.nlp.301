
# coding: utf-8

# In[ ]:


import glob
import json
import sys
import csv
import datetime
import traceback
import xml.etree.ElementTree as ET  

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report,roc_auc_score

import numpy as np

import nltk
nltk.download('punkt')

import pickle

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', '')

def print_time(*args):
    text = ""
    for arg in args:
        text += arg
    print("[",datetime.datetime.now(),"]", text)
    

def pickle_save(filename, data2pkl):
    global pickle_overwrite
    if pickle_overwrite == True:
        fileObject = open(filename,'wb') 
        pickle.dump(data2pkl,fileObject)
        fileObject.close()

        
def pickle_load(filename):
    fileObject = open(filename,'rb') 
    data_unpkl = pickle.load(fileObject)
    fileObject.close()
    return data_unpkl


# Logistic Regression model
def run_lr(dataset, label_set):
    split_num = int(len(dataset)*0.9)
    print("split num", split_num)
    train_set = dataset[:split_num]
    test_set = dataset[split_num:]
    train_label = label_set[:split_num]
    test_label = label_set[split_num:]
    
    print("training set", len(train_set),len(train_label),"testing set", len(test_set),len(test_label))
    
    print_time("Start LR...")

    lr = LogisticRegression(max_iter = 10) #, class_weight = {"0":.83, "1":.17}

    lr.fit(train_set, train_label)
    print_time("Complete LR...")
    
    lr_result = lr.predict(test_set)
    lr_probs = lr.predict_proba(test_set)
    
    lr_probs_trans = list(map(float,np.hstack(lr_probs[:,1])))

    print(classification_report(test_label, lr_result))
    print('ROC-AUC: ', roc_auc_score(test_label, lr_probs_trans))

    print(sum(list(map(int, lr_result))))
    
    
# display the keywords of each LDA topic    
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        word_idx = np.argsort(topic)[::-1][:no_top_words]
        
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# plot word cloud
def show_wordcloud(corpus):
    stopwords = set(STOPWORDS)

    # Create and generate a word cloud image:
    wordcloud = WordCloud(stopwords=stopwords).generate(corpus)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    



xml_file = "data/AviationData.xml"

narrative_path  = "data/*.json"

out_csv = "output/aviation_accidents.csv"

total_event_xml = 0

event_record_dict = {} # key: event, value: dict of attributes

# decide if need to pickle data    
pickle_overwrite = True


# In[ ]:


# process xml file
try:
    tree = ET.parse(xml_file)  
    root = tree.getroot()
    
    attr_dict = root[0][0].attrib
    
    for item in root:
            for subitem in item:
                event_json = subitem.attrib
                if total_event_xml == 1:                    
                    print(subitem.attrib)
                total_event_xml += 1
                event_record_dict[event_json.get("EventId")] = event_json
        
        
    with open(out_csv, mode='w') as outfile:
        writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(attr_dict.keys())

        for item in root:
            for subitem in item:
                if total_event_xml == 1:
                    print(subitem.attrib)
                writer.writerow(subitem.attrib.values())
                total_event_xml += 1
        
except Exception as ex:
    sys.stderr.write('Exception\n')
    extype, exvalue, extrace = sys.exc_info()
    traceback.print_exception(extype, exvalue, extrace)     
    


# In[ ]:


# process json files
event_nar_dict = {}

narrative_files = glob.glob(narrative_path)
total_event_json = 0

nar_corpus = []
cause_corpus = []
combined_corpus = []

for name in narrative_files:
    try:
        with open(name) as file:
            data = json.load(file).get("data")
            for event in data:
                event_id = event.get("EventId")
                
                event_nar_dict[event_id] = event                

                nar_corpus.append(event.get("narrative"))
                cause_corpus.append(event.get("probable_cause"))
                combined_corpus.append(event.get("narrative") + event.get("probable_cause"))
                
                if event_record_dict.get(event_id) != None:
                    record_dict = event_record_dict.get(event_id)
                    record_dict["narrative"] = event.get("narrative")
                    record_dict["probable_cause"] = event.get("probable_cause")
                    
                    svrt = record_dict.get("InjurySeverity")
                    if svrt[:5].lower() == "fatal":
                        record_dict["fatalness"] ="1"
                    else:
                        record_dict["fatalness"] ="0"
                else:
                    print("[ERR] Didn't find EventId in xml data.")
        
            total_event_json += len(data)
        
    except Exception as ex:
        sys.stderr.write('Exception\n')
        extype, exvalue, extrace = sys.exc_info()
        traceback.print_exception(extype, exvalue, extrace)
        


# In[ ]:


# preprocessing, filter out data without probable_cause for word cloud
data_set = []
label_set = []

fatal_corpus = []
nonfatal_corpus = []

for key, value in event_record_dict.items():
    cause = value.get("probable_cause")
    
    # preprocessing of the text, text = cause + narrative
    text = value.get("narrative") + value.get("probable_cause")

    if text[:4].upper() == "NTSB" or text[:21].lower() == "the foreign authority":
        text = text.split(".",1)[1].lower()
        text = text.replace("plt", "pilot").replace("flt", "flight")
        
    label = value.get("fatalness")
    
    if text != "":
        data_set.append(text)
        label_set.append(label)
        
    if cause != "":
        if label == "1":
            fatal_corpus.append(cause)
        else:
            nonfatal_corpus.append(cause)
        


# In[ ]:


# show word cloud for fatal/nonfatal events    
fatal_corpus_str = ' '.join(fatal_corpus)
show_wordcloud(fatal_corpus_str)    

nonfatal_corpus_str = ' '.join(nonfatal_corpus)
show_wordcloud(nonfatal_corpus_str)    


# In[ ]:


# topic modeling with sklearn LDA

no_features = 200
no_top_words = 10
no_topics = 10

from nltk.stem import PorterStemmer

porter = PorterStemmer()

data_set_stem = []

for item in data_set:    
    data_set_stem.append(porter.stem(item))


tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2),token_pattern=r'\b\w+\b', max_features=no_features, stop_words='english')

tf = tf_vectorizer.fit_transform(data_set_stem)

tf_feature_names = tf_vectorizer.get_feature_names()

print_time("Complete TF-IDF.")


# In[ ]:


print_time("Start LDA...")

lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)

lda_learner = lda.fit_transform(tf)

print_time("Complete LDA.")


# In[ ]:


# in case model run for too long, pickle data
# pickle_save("lda_learner_75590.pkl", lda_learner)
# pickle_save("lda_75590.pkl", lda)

# lda_learner = pickle_load("lda_learner_75590.pkl")
# lad = pickle_load("lda_75590.pkl")


# display_topics(lda, tf_feature_names, no_top_words)

x = {}

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx))
    topic_keywords = " ".join([tf_feature_names[i]
                    for i in topic.argsort()[:-no_top_words - 1:-1]])
    print(topic_keywords)
    topic_dict[topic_idx] = topic_keywords

topic_fatal = [0] * 10
topic_nonfatal = [0] * 10

for i,item in enumerate(lda_learner):   
    item = item.tolist()
    topic = item.index(max(item)) 
    if label_set[i] == "1":
        topic_fatal[topic] = topic_fatal[topic] + 1
    else:
        topic_nonfatal[topic] = topic_nonfatal[topic] + 1

for i in range(10):
    print(topic_fatal[i]/topic_nonfatal[i])


# In[ ]:


# run Logistic Regression and show metrics report
run_lr(lda_learner, label_set)


# In[ ]:


# plot a bar chart to show the fatal/nonfatal event counts in each topic
N = len(topic_dict)
ind = np.arange(N)  # the x locations for the groups
width = 0.3       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)


rects1 = ax.bar(ind, topic_fatal, width, color='r')

rects2 = ax.bar(ind+width, topic_nonfatal, width, color='g')

x_labels = []
for value in topic_dict.values():
    x_labels.append(value.replace(" s","").replace(" ", "\n"))

ax.set_ylabel('Scores')
ax.set_xticks(ind+width)
ax.set_xticklabels(x_labels)
ax.legend( (rects1[0], rects2[0]), ('fatal', 'nonfatal') )

plt.show()

