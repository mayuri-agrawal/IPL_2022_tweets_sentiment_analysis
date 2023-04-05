#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud 
import re  # for searching common words in a string
import os


# # Installing vader sentiment analyser

# In[2]:


get_ipython().system('pip install vaderSentiment')


# # Importing Vader sentiment analyzer
# 

# In[3]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# # Importing Dataset

# In[4]:


df = pd.read_csv ('IPL_2022_tweets.csv')
df.head()


# # Analysing the dataset

# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# # Copying text to perform EDA

# In[8]:


df['senttext'] = df['text']


# # Conterting all string data to lowercase

# In[9]:


df = df.apply(lambda x: x.astype(str).str.lower())


# In[10]:


loc_df = df[df.user_location != 'nan']
loc_df.user_location.value_counts().nlargest(20).plot(kind='bar',figsize=(25,10))


# In[11]:


indian_cities = {}
indian_ipl_cities = ['mumbai','bangalore','chennai','delhi','kolkata','lucknow','ahmedabad','hyderabad','punjab','jaipur']
for city in indian_ipl_cities:
    indian_cities[city] = df.user_location.str.count(city).sum()
    
plt.figure(figsize=(25,10))
plt.bar(*zip(*indian_cities.items()))
plt.show


# Checking if the user account is verified or not

# In[12]:


df.user_verified.value_counts().nlargest(2).plot(kind='bar')


# This shows that most of the users are not verified

# # Checking most commonly using hastags

# In[13]:


hashtag_df = df[df.hashtags != 'nan']
hashtag_df.hashtags.value_counts().nlargest(5).plot(kind='bar', rot=0, figsize=(20,8))


# #ipl2022 is most commonly used hashtag

# In[14]:


import nltk
nltk.download('stopwords')


# #  Preprocessing the data

# In[15]:


from nltk.corpus import stopwords

stop_words = stopwords.words('english')
df.text = df.text.apply(lambda x:' '.join([word for word in x.split() if word not in (stop_words)]))


# In[16]:


df.text = df.text.apply(lambda x: ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", x).split()))


# In[17]:


df.text = df.text.apply(lambda x: ' '.join(re.sub("[\.\,\!\?\:\;\-\=\_\'\*\"|(|)]", " ", x).split()))


# In[18]:


df.text = df.text.apply(lambda x: ' '.join(re.sub(r'http\S+', '',x).split()))


# In[19]:


df.text.head()


# Data is cleaned except emojis

# # Creating a wordcloud

# In[20]:


wordcloud = WordCloud(
                          background_color='white',
                          colormap='Reds',
                          max_words=200,
                          max_font_size=40, 
                          random_state=49
                         ).generate(str(df['text']))

plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# # Replacing the emojis

# In[31]:


try:
    # UCS-4
    e = re.compile(u'[\U00010000-\U0010ffff]')
except re.error:
    # UCS-2
    e = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
emojis = []
for x in df.text:
    match  = e.search(x)
    if match:
        emojis.append(match.group())


# In[32]:


dfe =  pd.DataFrame(emojis,columns=['text'])
pd.Series(' '.join(dfe['text']).lower().split()).value_counts()[:10]


# # Finding similar words using word2vec

# In[33]:


num_features = 400    # Word vector dimensionality                      
min_word_count = 5   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

wt = [list(x.split()) for x in df.text]
from gensim.models import word2vec
print ("Training model...")
wv_model = word2vec.Word2Vec(wt, workers=num_workers,             vector_size=num_features, min_count = min_word_count,             window = context, sample = downsampling)

wv_model.init_sims(replace=True)


# In[34]:


wv_model.wv.most_similar("ipl") 


# In[35]:


wv_model.wv.most_similar("dhoni") 


# In[36]:


wv_model.wv.most_similar("captain") 


# # Now applying VADER sentiment analyser

# In[37]:


analyser = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['senttext'].apply(lambda x: analyser.polarity_scores(str(x)))


# In[38]:


def sentiment_func(sentiment):
    for k,v in sentiment.items():
        if (k== 'pos' or k or 'neg' or k == 'neu') == True:
            if (sentiment['pos'] > 0.5 and sentiment['neg'] < 0.5 and sentiment['neu'] < 0.5) == True:
                return 'positive'
            elif (sentiment['pos'] < 0.5 and sentiment['neg'] > 0.5 and sentiment['neu'] < 0.5) == True:
                return 'negative'
            elif (sentiment['pos'] < 0.5 and sentiment['neg'] < 0.5 and sentiment['neu'] > 0.5) == True:
                return 'neutral'

df['sentiment'] = df['sentiment_score'].apply(sentiment_func)


# In[39]:


df.sentiment.value_counts().plot(kind='bar', rot=0)


# In[40]:


df.sentiment.value_counts()


# Most the tweets are neutral. This can be due to most tweets just containing score updates or match updates.
# 
# Number of positive tweets are more than negative. Seems like people were very happy with tournament happening at such difficult time and people got excited and happy to see their favorite cricketers back on pitch.

# In[45]:


nltk.download('vader_lexicon')


# In[5]:


val = input("Enter your value: ")


# In[6]:



from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser2 = SentimentIntensityAnalyzer()
sentiment = analyser2.polarity_scores(val)
sentiment


# In[ ]:




