from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Embedding, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.callbacks import History
import pickle

#-------------------------- Load WordVectors from PreTrained Data ------------------------------
embeddings_index = {}
errorCount = 0

f = open('E:\Word2Vec\wiki.en.vec','r',encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	try:
		coefs = np.asarray(values[1:], dtype='float32')
	except ValueError:
		errorCount+=1
		#print("Error on line ",line)
	embeddings_index[word] = coefs
f.close()

EMBEDDING_DIM = len(values[1:])
print('Found %s word vectors.' % len(embeddings_index))
print('Error count: '+str(errorCount))
#-----------------------------------------------------------------------------------------------
load_state_from_variable = -1
		
		
def tokenize(s):
	#return [c for c in ' '.join(s.split())]
	all_words = word_tokenize(s)
	lexicon = [i for i in all_words if len(i)>1] #all_words #[lemmatizer.lemmatize(i) for i in all_words]
	return lexicon

path = 'E:\Articles\Titles.tsv' #'E:\Articles\AllArticleTitles.tsv' #'E:\Articles\Titles.tsv' #get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
#path = 'E:\Articles\Top6MDistinctAdDescriptions.tsv'

lemmatizer = WordNetLemmatizer()
text = tokenize(open(path).read().lower())
print('corpus length:', len(text))
chars = sorted(list(set(text)))

boundaries = [ i for i,x in enumerate(text) if x =="eeeee" ]
allsen = [text[(boundaries[i]+1):(boundaries[i+1]+1)] for i in range(len(boundaries)-2)]
text = []
[text.extend(allsen[int((len(allsen)-1)*np.random.uniform())]) for i in range(len(allsen))]

print('total Words:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

embedding_matrix = np.zeros((len(char_indices) + 1, EMBEDDING_DIM))
for word, i in char_indices.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector
	else:
		embedding_matrix[i] = np.ones(EMBEDDING_DIM)*i

# cut the text in semi-redundant sequences of maxlen words
maxlen = 5
step = 5
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
	sentences.append(text[i: i + maxlen])
	next_chars.append(text[i + maxlen])

print('nb sequences:', len(sentences))
print('Vectorization...')

X = np.zeros((len(sentences), maxlen), dtype=np.float32)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		X[i, t] = char_indices[char]
	y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Embedding(len(chars)+1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=maxlen, trainable=False))
#model.add(Conv1D(128, 5, activation='relu'))
#model.add(MaxPooling1D(5))
model.add(Dropout(0.05))
model.add(LSTM(128))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])





def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

saturated = False;

model.load_weights('iteration_1.h5')
'''
# train the model, output generated text after each iteration
prev_loss = 1000000.0
for iteration in range(1, 2):
	print()
	print('-' * 50)
	print('Iteration', iteration)
	
	if(saturated):
		print("Saturated - regenerating training sequences")
		text = []
		[text.extend(allsen[int((len(allsen)-1)*np.random.uniform())]) for i in range(len(allsen))]
		sentences = []
		next_chars = []

		for i in range(0, len(text) - maxlen, step):
			sentences.append(text[i: i + maxlen])
			next_chars.append(text[i + maxlen])

		print('nb sequences:', len(sentences))
		print('Vectorization...')
		X = np.zeros((len(sentences), maxlen), dtype=np.float32)
		y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
		for i, sentence in enumerate(sentences):
			for t, char in enumerate(sentence):
				X[i, t] = char_indices[char]
			y[i, char_indices[next_chars[i]]] = 1

	if(load_state_from_variable>0):
		model.load_weights('iteration_'+str(load_state_from_variable)+'.h5')
	
	hist = model.fit(X, y,
			  batch_size=512,
			  epochs=50)

	if(hist.history['loss'][-1]<=prev_loss):
		model.save_weights('iteration_'+str(iteration)+'.h5')
		prev_loss = hist.history['loss'][-1]
		countBad = 0
	else:
		model.load_weights('iteration_'+str(iteration-1)+'.h5')
		model.save_weights('iteration_'+str(iteration)+'.h5')
		countBad += 1
		if(countBad>4):
			saturated=True
			countBad = -5

			
	model.save_weights('iteration_'+str(iteration)+'.h5')
	
	start_index = random.randint(0, len(text) - maxlen - 1)
'''
'''
	for diversity in [0.02, 0.2, 0.5]:
		#print()
		#print('----- diversity:', diversity)

		generated = ''
		sentence = text[start_index: start_index + maxlen]
		generated += ' '.join(sentence)
		#print('----- Generating with seed: "' + ' '.join(sentence) + '"')
		#sys.stdout.write(generated)

		for i in range(50):
			x = np.zeros((1, maxlen))
			for t, char in enumerate(sentence):
				x[0, t] = char_indices[char]

			preds = model.predict(x, verbose=0)[0]
			next_index = sample(preds, diversity)
			next_char = indices_char[next_index]

			generated += ' '+next_char
			sentence = sentence[1:] + [next_char]

			#sys.stdout.write(' '+next_char)
			#sys.stdout.flush()
		#print()
'''
'''
with open('ModelData.txt','w') as f:
	pickle.dump(model,f)
	pickle.dump(sentences,f)
	pickle.dump(char_indices,f)
	pickle.dump(indices_char,f)
	pickle.dump(text,f)

allsentences = []
with open('GeneratedSentences.txt','w') as f:
	for diversity in [0.02, 0.1, 0.2, 0.5]:
		#print()
		#print('----- diversity:', diversity)
		f.write("Diversity:"+str(diversity)+":\n")
		for j in range(100):
			start_index = random.randint(0, len(text) - maxlen - 1)
			generated = ''
			sentence = text[start_index: start_index + maxlen]
			generated += ' '.join(sentence)
			#print('----- Generating with seed: "' + ' '.join(sentence) + '"')
			sys.stdout.write(generated)
			
			#f.write("Generated:\n"+generated)
			for i in range(2000):
				x = np.zeros((1, maxlen))
				for t, char in enumerate(sentence):
					x[0, t] = char_indices[char]

				preds = model.predict(x, verbose=0)[0]
				next_index = sample(preds, diversity)
				next_char = indices_char[next_index]

				generated += ' '+next_char
				sentence = sentence[1:] + [next_char]
				allsentences.extend(next_char)

				#sys.stdout.write(' '+next_char)
				f.write(' '+next_char)
				#sys.stdout.flush()
				f.flush()
			print()
		
'''
'''
allsentences = []
bp = 0.05
rs = False
with open('GeneratedSentences_NM.txt','w') as f:
	for diversity in [0.02, 0.1, 0.2, 0.5]:
		#print()
		#print('----- diversity:', diversity)
		f.write("Diversity:"+str(diversity)+":\n")
		for j in range(100):
			start_index = random.randint(0, len(text) - maxlen - 1)
			generated = ''
			sentence = text[start_index: start_index + maxlen]
			generated += ' '.join(sentence)
			#print('----- Generating with seed: "' + ' '.join(sentence) + '"')
			sys.stdout.write(generated)
			
			#f.write("Generated:\n"+generated)
			for i in range(2000):
				x = np.zeros((1, maxlen))
				for t, char in enumerate(sentence):
					x[0, t] = char_indices[char]

				preds = model.predict(x, verbose=0)[0]
				if(random.random() < bp):
					rs = True
					next_index = sample(preds, 1)
				else:
					next_index = sample(preds, diversity)
				next_char = indices_char[next_index]

				generated += ' '+next_char
				sentence = sentence[1:] + [next_char]
				allsentences.extend(next_char)

				#sys.stdout.write(' '+next_char)
				if(rs):
					f.write(' #R#'+next_char)
					rs = False
				else:
					f.write(' '+next_char)
				#sys.stdout.flush()
				f.flush()
			print()
		
'''
sys.stdout.write("(1) Search Gen Set:")
sys.stdout.flush()
for i in range(1000):
	input_str = sys.stdin.readline()
	sentence = tokenize(input_str)
	if(sentence[0] in char_indices):
		occs = [i for i, v in enumerate(text) if v == sentence[0]]
		sensbl = [max([j for j in boundaries if j<i]) for i in occs]
		subb = [[i for i, v in enumerate(boundaries) if v == j][0] for j in sensbl]
		sens = [text[(boundaries[i]+1):(boundaries[i+1]+1)] for i in subb if i+1 < len(boundaries)]
		for sent in sens:
			broken_sent = sent
			for t, char in enumerate(sent):
				if(random.random()<0.2):
					broken_sent[t] = 'xxxxx'
			bs = ['eeeee','eeeee','eeeee','eeeee','eeeee']
			bs.extend(broken_sent)
			broken_sent = bs
			sentence = broken_sent[0:5]
			for t, char in enumerate(broken_sent):
				if((char == 'xxxxx') & (len(sentence)==maxlen)):
					x = np.zeros((1, maxlen))
					for t, char in enumerate(sentence):
						if(char in char_indices):
							x[0, t] = char_indices[char]
						else:
							x[0,t] = 0
						
					preds = model.predict(x, verbose=0)[0]
					next_index = sample(preds, 0.002)
					next_char = indices_char[next_index]
					sys.stdout.write(' '+next_char)
					sys.stdout.flush()
					if (next_char == 'eeeee'):
						break;
					sentence = sentence[1:] + [next_char]
				else:
					if(len(sentence)==maxlen):
						sentence = sentence[1:]+[char]
					else:
						sentence = sentence+[char]
					sys.stdout.write(' '+char)
					sys.stdout.flush()
			print()
		print('Next Word:')
		sys.stdout.flush()
sys.stdout.write("(2) Manual Input Mode:")
sys.stdout.flush()
for i in range(1000):
	input_str = sys.stdin.readline()
	sentence = tokenize(input_str)
	for j in range(500):
		x = np.zeros((1, maxlen))
		for t, char in enumerate(sentence):
			if(t > maxlen):
				break
			if(char in char_indices):
				x[0, t] = char_indices[char]
			else:
				x[0,t] = 0
		preds = model.predict(x, verbose=0)[0]
		next_index = sample(preds, 0.002)
		next_char = indices_char[next_index]

		if (next_char == 'eeeee'):
			break;
		sentence = sentence[1:] + [next_char]

		sys.stdout.write(' '+next_char)
		sys.stdout.flush()
	print()
	
	
'''
text = tokenize(allsentences)
boundaries = [ i for i,x in enumerate(text) if x =="eeeee"]
with open('GeneratedSentences.txt','w') as f:
	for i in range(0, len(boundaries)-1):
		baseSentence = text[(boundaries[i]+1): ((boundaries[i+1])+1)]
		f.writeline(sum(baseSentence))
		f.flush()

Looks good - but need to ensure the input vector sequence has enough history for the model to predict well : 
	GAN?
	[x] Train on more data?
	Restrict Vertical?
	
'''