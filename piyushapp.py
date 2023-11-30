import streamlit as st
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
import pickle
import numpy as np

model_here = pickle.load(open('piyushmodel.pkl','rb'))
data_here = pickle.load(open('data.pkl','rb'))

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data_here['text'].values)

image = Image.open('mitaoe-logo.jpg')
st.image(image)
st.header('Hello!! from - Twitter Sentix - MITAOE')
st.header('Add the tweet or message here, we will analyze for you !!!')

msg_tweet = st.text_input('Paste text')
temp = []
temp.append(msg_tweet)
msg_tweet = temp
msg_tweet = tokenizer.texts_to_sequences(msg_tweet)
msg_tweet = pad_sequences(msg_tweet, maxlen=28, dtype='int32', value=0)

if st.button('Analyze'):
    sentiment_here = model_here.predict(msg_tweet,batch_size=1,verbose = 2)[0]
    result=np.argmax(sentiment_here)

    if result == 0:
        st.header('Negative')
    else:
        st.header('Positive')

st.header('Team Member: ')
st.text('05 - Piyush Bhondave')
st.text('66 - Rutuja Mahajan')
st.text('78 - Akansha Madamanchi')
