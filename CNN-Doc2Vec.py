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
from keras.layers import Conv1D, MaxPooling1D,Flatten

    
wb_X_train = np.array(wb_X_train)
wb_X_test = np.array(wb_X_test)
wb_Y_train = np.array(wb_Y_train)
wb_Y_test = np.array(wb_Y_test)
print wb_X_train.shape
wb_X_train = wb_X_train.reshape(wb_X_train.shape[0],wb_X_train.shape[1],1)
print wb_X_train.shape
model = Sequential()    
model.add(Conv1D(filters=500, kernel_size=2, activation='relu', strides=1, input_shape=(wb_X_train.shape[1],1)))
# we use max pooling:
#model.add(Activation('relu'))
model.add(MaxPooling1D())
#model.add(Flatten())

# We add a vanilla hidden layer:
#model.add(Dense(70))
# model.add(Conv1D(filters=300, kernel_size=2, strides=1))
# model.add(Activation('relu'))
# model.add(MaxPooling1D())

model.add(Conv1D(filters=100, kernel_size=2, strides=1))
model.add(Activation('relu'))
model.add(MaxPooling1D())



model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
#model.fit(wb_X_train, wb_Y_train, batch_size=32,
#          epochs=10)
wb_X_test = wb_X_test.reshape(wb_X_test.shape[0],wb_X_test.shape[1],1)
model.fit(wb_X_train, wb_Y_train, batch_size=32,
          epochs=20, validation_data=(wb_X_test, wb_Y_test))

scores = model.evaluate(wb_X_test, wb_Y_test, verbose=0)
print('CNN test score:', scores[0])
print('CNN test accuracy:', scores[1])

y_pred = model.predict(wb_X_test, verbose=1)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(wb_Y_test.argmax(1), y_pred.argmax(1))
# print cm

from sklearn.metrics import accuracy_score

#print(wb_Y_test)
# print(np.shape(wb_Y_test), np.shape(y_pred))
print("Accuracy : ", accuracy_score(wb_Y_test.argmax(1), y_pred.argmax(1), normalize=False)/float(len(wb_Y_test)))


#from sklearn.metrics import classification_report
