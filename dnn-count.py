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


seed = 7
np.random.seed(seed)


dataset = pandas.read_csv('./merged-data.txt',header = None, delimiter="\t", encoding='utf-8')


dataset = dataset.drop(1, 1)
dict = {'DEMO': 1, 'DISE': 2, 'TRMT': 3,'GOAL': 4, 'PREG': 5, 'FAML': 6,'SOCL': 7}
for index, row in dataset.iterrows():
	row[0]=dict[row[0]]
X = dataset[2]
Y = dataset[0]
#one hot encoded
y = np_utils.to_categorical(Y)

print y[0]


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
x_train = count_vect.fit_transform(X)
print x_train.shape


#PCA 
print 'starting PCA'
svd = TruncatedSVD(n_components=500, n_iter=7, random_state=42)
svd.fit(x_train)
x_train = svd.transform(x_train)
print 'PCA Processing complete'


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, x_train, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#estimator.save('my_dnn_model.h5')





