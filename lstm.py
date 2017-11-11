import numpy as np
import pandas
import os
from glob import glob
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# dataset = pandas.read_csv('ICHI2016-TrainData.tsv', usecols=[1], engine='python', skipfooter=3)
dataset = pandas.read_csv('input-data.txt',header = None, delimiter="\t", encoding='utf-8')
# dataset =DataFrame.from_csv('ICHI2016-TrainData.tsv', sep='\t', header=0)

dataset = dataset.drop(1, 1)
dict = {'DEMO': 1, 'DISE': 2, 'TRMT': 3,'GOAL': 4, 'PREG': 5, 'FAML': 6,'SOCL': 7}
for index, row in dataset.iterrows():
	row[0]=dict[row[0]]
X = dataset[2]
Y = dataset[0]
#one hot encoded
y = np_utils.to_categorical(Y)

print y[0]
# convert feature into tfidf model
# first bag of word

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X)
print X_train_counts.shape

# tfidf model

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print X_train_tfidf.shape
X_train_tfidf = X_train_tfidf.astype('float32')
# split data into training and testing
print 'PCA start'
svd = TruncatedSVD(n_components=500, n_iter=7, random_state=42)
svd.fit(X_train_tfidf)
X_train_tfidf = svd.transform(X_train_tfidf)
print 'PCA done'


X_train , X_test , y_train ,y_test= train_test_split(X_train_tfidf,y,test_size=0.4)



#lstm model

# create the model

embedding_vecor_length = 500
max_review_length= 500
top_words=8000
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(8, activation='softmax'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)



# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'



# load model 
# model = load_model('my_model.h5')
