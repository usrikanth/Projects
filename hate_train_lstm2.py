<<<<<<< HEAD

# coding: utf-8

# In[42]:


import csv
import numpy as np
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

#max number of unique words should be in the order of 20 - 50K using 5000 here

vocab_size = 5000

#maximum number of words in a query/X

max_len = 8

#Read data
def genData():
    #X, Y arrays for all data
    X = []
    Y = []

    #generate 20% as test set
    train_count=0
    validate_count = 0
    test_count = 0
    line_count = 0

    with open('./hs.csv', encoding='latin-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            X.append(row[0])
            Y.append(row[1])
            line_count += 1
          

        #encode the vocab
        encoded_X = [one_hot(d, vocab_size) for d in X]
        padded_X = pad_sequences(encoded_X, maxlen=max_len, padding="post")
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        Y_one_hot = np_utils.to_categorical(encoded_Y)

        #Training, Test
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        # generate sets for train, test and validate
        np.random.seed(1)
        for i in range(line_count):
            if(round(np.random.rand()*100) < 81):
                X_train.append(padded_X[i])
                Y_train.append(Y_one_hot[i])
                train_count = train_count+1
            else:
                X_test.append(padded_X[i])
                Y_test.append(Y_one_hot[i])
                test_count = test_count+1
    return X_train, Y_train, X_test, Y_test


# In[43]:


#main code 
#get data
X_train, Y_train, X_test, Y_test = genData()
print(X_train[10],", ", Y_train[10])


# In[47]:


#build model



encode_dim = 32
model = Sequential()
model.add(Embedding(vocab_size, encode_dim, input_length=max_len))

#LSTM layer size

hidden_units_size = 100
model.add(LSTM(hidden_units_size, return_sequences=False))
model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print(model.summary())


# In[45]:


#run the model

# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

=======
import csv
import numpy as np
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# define documents

#max number of unique words should be in the order of 20 - 50K using 5000 here
vocab_size = 5000
#maximum number of words in a query/X
max_len = 8


def genData():
    #X, Y arrays for all data
    X = []
    Y = []
    

    #generate 20% as test set
    train_count=0
    validate_count = 0
    test_count = 0
    line_count = 0

    with open('hs.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            X.append(row[0])
            Y.append(row[1])
            line_count += 1
            

        #encode the vocab
        encoded_X = [one_hot(d, vocab_size) for d in X]
        padded_X = pad_sequences(encoded_X, maxlen=max_len, padding="post")

        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        Y_one_hot = np_utils.to_categorical(encoded_Y)
        

        #Training, Test
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        # generate sets for train, test and validate
        np.random.seed(1)
        for i in range(line_count):
            if(round(np.random.rand()*100) < 81):
                X_train.append(padded_X[i])
                Y_train.append(Y_one_hot[i])
                train_count = train_count+1
            else:
                X_test.append(padded_X[i])
                Y_test.append(Y_one_hot[i])
                test_count = test_count+1
                
    return X_train, Y_train, X_test, Y_test

#main code 

#get data
X_train, Y_train, X_test, Y_test = genData()

#build the DNN/LSTM

encode_dim = 32


#build model
model = Sequential()
model.add(Embedding(vocab_size, encode_dim, input_length=max_len))
#LSTM layer size
hidden_size = 100
model.add(LSTM(hidden_size, return_sequences=False))
model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit
>>>>>>> 48db496b23ea07f6e8cf22d3170285335884d071
