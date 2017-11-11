import numpy as np
import pandas
import os
from glob import glob
import matplotlib.pyplot as plt

from keras.models import Model
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=500, activation='relu'))
	model.add(Dense(8, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

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


# #auto encoder
# # sX = minmax_scale(X_train_tfidf, axis = 0)
# sX=X_train_tfidf
# X_train, X_test, Y_train, Y_test = train_test_split(sX, y, train_size = 0.5)
# ncol = sX.shape[1]
# print ncol
# input_dim = Input(shape = (ncol, ))
# # DEFINE THE DIMENSION OF ENCODER ASSUMED 500
# encoding_dim = 50
# # DEFINE THE ENCODER LAYER
# encoded = Dense(encoding_dim, activation = 'relu')(input_dim)
# # DEFINE THE DECODER LAYER
# decoded = Dense(ncol, activation = 'sigmoid')(encoded)
# # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
# autoencoder = Model(input = input_dim, output = decoded)
# # CONFIGURE AND TRAIN THE AUTOENCODER
# autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
# autoencoder.fit(X_train, X_train, nb_epoch = 50, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))
# # THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
# encoder = Model(input = input_dim, output = encoded)
# encoded_input = Input(shape = (encoding_dim, ))
# X_test_encoded = encoder.predict(sX)
# encoder.save('my_encoder.h5')

#PCA 
print 'PCA start'
svd = TruncatedSVD(n_components=500, n_iter=7, random_state=42)
svd.fit(X_train_tfidf)
X_train_tfidf = svd.transform(X_train_tfidf)
print 'PCA done'

# split data into training and testing

# from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split
# X_train , X_test , y_train ,y_test= train_test_split(X_train_tfidf,y,test_size=0.5)

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X_train_tfidf, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

estimator.save('my_dnn_model.h5')





