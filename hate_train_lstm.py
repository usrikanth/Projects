import csv
import numpy as np
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
# define documents

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
        vocab = list(set(X+Y))
        vocab_size = len(vocab)
        #Training, Test
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        for i in range(line_count):
            if(round(np.random.rand()*100) < 81):
                X_train.append(X[i])
                Y_train.append(Y[i])
                train_count = train_count+1
            else:
                X_test.append(X[i])
                Y_test.append(Y[i])
                test_count = test_count+1
                
    return X, X_train, Y_train, X_test, Y_test, vocab_size, vocab

#main code 

#get data
X_train, Y_train, X_test, Y_test, vocab_size, vocab = genData()

#encode the vocab
encoded_vocab = [one_hot(d, vocab_size) for d in vocab]

encode_dim = 32

#max number of words in a sentence
max_words = 6
#pad the sentence length
padded_words = padded_sequences(encoded_vocab, maxlen=max_words, padding='post')

#build model
model = Sequential()
model.add(Embedding(embedded_vocab, encode_dim, input_length=max_words))
#LSTM layer size
hidden_size = 100
model.add(LSTM(hidden_size, return_sequences=False))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])


