# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.utils import np_utils

## Plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('hs.csv',encoding='latin-1',sep='\t', names=['text', 'label'])

#if 'clean' then 0 if 'adult', 'mature', 'racy' then 1
#labels = df['label'].map(lambda x:0 if (x.lower()=='clean') else 1)

#encode actual one hot for clean, adult, mature, racy etc.

Y = df['label']
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
labels = np_utils.to_categorical(encoded_Y)

print("labels - shape: ",labels.shape)

#create the data for consumption
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=50)

## Network architecture
model = Sequential()
model.add(Embedding(20000, 100, input_length=50))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
print("data.shape: ",data.shape,' labels.shape: ',labels.shape)
## Fit the model
#model.fit(data, np.array(labels), validation_split=0.4, epochs=10)

