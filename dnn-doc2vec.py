import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from collections import namedtuple


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

from keras.utils import np_utils
Y = np_utils.to_categorical(Y)
docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(X):
	words = text.lower().split()
	tags = [i]
	docs.append(analyzedDocument(words, tags))
    
    


from gensim.models import doc2vec

model = doc2vec.Doc2Vec(docs,size=1000, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate


Y_train,Y_test=Y[:9000],Y[9000:]

wb_Y_train,wb_Y_test=Y_train,Y_test
wb_X=[]
for i in range(len(X)):
	wb_X.append(model.docvecs[i])
	wb_X_train=wb_X[:9000]
	wb_X_test=wb_X[9000:]


from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
# from keras.layers import Embedding
    
wb_X_train = np.array(wb_X_train)
wb_X_test = np.array(wb_X_test)
wb_Y_train = np.array(wb_Y_train)
wb_Y_test = np.array(wb_Y_test)
print wb_X_train.shape
print wb_X_train.shape


model = Sequential()
model.add(Dense(500, input_dim=1000, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(wb_X_train, wb_Y_train, epochs=30, batch_size=40)
scores = model.evaluate(wb_X_test, wb_Y_test, verbose=1)
print('DNN test score:', scores[0])
print('DNN test accuracy:', scores[1])