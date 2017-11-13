import numpy as np
#from sklearn.metrics import accuracy_score
#from sklearn import linear_model
#from collections import namedtuple


def label_to_int(label):
	labels={'SOCL':0,'PREG':1,'GOAL':2,'TRMT':3,'DEMO':4,'FAML':5,'DISE':6}
	return labels[label]


X=[]
Y=[]

with open('./input-data-merged.txt') as f:
	for line in f:
		line = line.split('\t\t')
		X.append(line[1])
		Y.append(label_to_int(line[0]))



from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#Do some very minor text preprocessing
def cleanText(corpus):
    corpus = [z.lower().replace('\n','').split() for z in corpus]
    return corpus

x_train = cleanText(x_train)
x_test = cleanText(x_test)


def buildWordVector(text, size):
	vec = np.zeros(size).reshape((1, size))
	count = 0.
	for word in text:
		try:
			vec += model[word].reshape((1, size))
			count += 1.
		except KeyError:
			continue
		if count != 0:
			vec /= count
		return vec



from gensim.models.word2vec import Word2Vec
n_dim = 1000
#Initialize model and build vocab
model = Word2Vec(size=n_dim, min_count=1)
model.build_vocab(x_train)

#Train the model over train_reviews (this may take several minutes)
model.train(x_train,total_examples=model.corpus_count,epochs=10)


#from sklearn.preprocessing import scale
train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
#train_vecs = scale(train_vecs)

#Train word2vec on test tweets
model.train(x_test,total_examples=model.corpus_count,epochs=10)

#Build test tweet vectors then scale
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
#test_vecs = scale(x_test)

train_vecs = np.array(train_vecs)
test_vecs = np.array(test_vecs)
y_train = np.array(y_train)
y_test = np.array(y_test)


from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(500, input_dim=1000, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_vecs, y_train, epochs=30, batch_size=40)
scores = model.evaluate(test_vecs, y_test, verbose=1)
print('DNN test score:', scores[0])
print('DNN test accuracy:', scores[1])
