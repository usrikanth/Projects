
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
from keras.utils.vis_utils import plot_model
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
        print("actual vocab size",len(set(X)))  
        output_size = len(set(Y))
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
    return X_train, Y_train, X_test, Y_test, output_size


# In[43]:


#main code 
#get data
X_train, Y_train, X_test, Y_test, output_size = genData()
print("Sample X and Y: ",X_train[10],", ", Y_train[10])
print("output size ", output_size)


# In[47]:


#build model

encode_dim = 32
model = Sequential(max_len)

#add embedding layer
model.add(Embedding(vocab_size, encode_dim, input_length=max_len))

#add LSTM layer

hidden_units_size = 40
model.add(LSTM(hidden_units_size, return_sequences=False))


model.add(Dense(output_size, activation='softmax'))

#compile the model with categorical cross entropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print(model.summary())


# In[45]:


#run the model
'''
# fit the model
model.fit(X_train, Y_train, epochs=5, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))

'''
