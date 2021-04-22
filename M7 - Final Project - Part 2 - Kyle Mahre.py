# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:33:59 2019

@author: kfmah
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
#from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import datetime

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df = pd.read_csv("tmdb-movie-metadata/tmdb_5000_movies.csv", delimiter = ",", header = 0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def midbudge(budget):
    if (budget < 85000000 and budget >= 30000000):
        return True
    else:
        return False
df = df.drop_duplicates(['original_title'])
df = df[['id','original_title','budget', 'genres', 'release_date', 'revenue', 'runtime', 'popularity', 'vote_average','vote_count']].dropna()
df['budget'].dropna()
df['revenue'].dropna()
df.drop(df.loc[df['budget']==0].index, inplace=True)
df.drop(df.loc[df['revenue']==0].index, inplace=True)
df['profit'] = df['revenue']-df['budget']

# three binary categorical classes for budget

df['low_budget'] = df['budget']<30000000
df['low_budget'] = df['low_budget'].astype(int)
df['mid_budget'] = df['budget'].apply(midbudge)
df['mid_budget'] = df['mid_budget'].astype(int)
df['big_budget'] = df['budget']>=85000000
df['big_budget'] = df['big_budget'].astype(int)

# six binary categorical classes for profit

df['flop'] = df['profit']<0
df['flop'] = df['flop'].astype(int)

df['bomb'] = df['profit']<.75*df['budget']
df['bomb'] = df['bomb'].astype(int)

df['mediocre'] = df['profit']>=.75*df['budget']
df['mediocre'] = df['mediocre'].astype(int)

df['success'] = df['profit']>1.5*df['budget']
df['success'] = df['success'].astype(int)

df['blockbuster'] = df['profit']>2.5*df['budget']
df['blockbuster'] = df['blockbuster'].astype(int)

df['super_hit'] = df['profit']>4*df['budget']
df['super_hit'] = df['super_hit'].astype(int)

df_genre = pd.DataFrame(columns = ['id','original_title','genre', 'cgenres', 'low_budget', 'mid_budget', 'big_budget', 'revenue', 'day', 'month', 'year', 'runtime', 'popularity', 'vote_average', 'vote_count', 'flop', 'bomb', 'mediocre', 'success', 'blockbuster', 'super_hit'])

# pulls out all the genres, makes a new row for each - aggregated at 147

def dataPrep(row):
    global df_genre
    d = {}
    genres = np.array([g['name'] for g in eval(row['genres'])])
    n = genres.size
    d['id'] = [row['id']]*n
    d['original_title'] = [row['original_title']]*n
    d['low_budget'] = [row['low_budget']]*n
    d['mid_budget'] = [row['mid_budget']]*n
    d['big_budget'] = [row['big_budget']]*n
    d['runtime'] = [row['runtime']]*n
    d['popularity'] = [row['popularity']]*n
    d['vote_average'] = [row['vote_average']]*n
    d['vote_count'] = [row['vote_count']]*n
    d['revenue'] = [row['revenue']]*n
    d['flop'] = [row['flop']]*n
    d['bomb'] = [row['bomb']]*n
    d['mediocre'] = [row['mediocre']]*n
    d['success'] = [row['success']]*n
    d['blockbuster'] = [row['blockbuster']]*n
    d['super_hit'] = [row['super_hit']]*n
    d.update(zip(('year', 'month', 'day'), map(int, row['release_date'].split('-'))))
    d['genre'], d['cgenres'] = [], []
    for genre in genres:
        d['genre'].append(genre)
        d['cgenres'].append(genres[genres != genre])
    df_genre = df_genre.append(pd.DataFrame(d), ignore_index=True, sort=True)  

df.apply(dataPrep, axis=1)
df_genre = df_genre[['id','original_title','genre', 'low_budget', 'mid_budget', 'big_budget', 'revenue', 'day', 'month', 'year', 'cgenres', 'runtime', 'popularity', 'vote_average', 'vote_count', 'flop', 'bomb', 'mediocre', 'success', 'blockbuster', 'super_hit']]
df_genre = df_genre.infer_objects()

# 114 - 142 makes a list an counts the instances of all the genre categories and adds a column to our data for it
def unique(list1): 
    x = np.array(list1) 
    y = np.unique(x)
    z = y.tolist()
    print(z)
    return z

le = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

genre_catagories = unique(df_genre['genre'])
integer_encoded = le.fit_transform(df_genre.loc[:,'genre']) 
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
genrecats = pd.DataFrame(onehot_encoded, columns = [genre_catagories])
genrecats_sums = []

for i in range(0,len(genre_catagories)):
    genrecats_sums.append(sum(genrecats[(genre_catagories[i]),]))
    
for i in range(0,len(genrecats.columns)):
    newcolumn = genrecats[(genre_catagories[i],)]
    name_tail = genre_catagories[i]
    name_tail = name_tail.upper()
    name_tail = name_tail.split(' ')
    name_tail = '_'.join(name_tail)
    df['GENRE_'+name_tail] = newcolumn
    df_genre['GENRE_'+name_tail] = newcolumn

# output a count of genres   
#genrecats.to_csv('Genre_categories.csv')

# aggregates the genres, grouped by original title
df_genre = df_genre.groupby('original_title').agg({'id':'first','low_budget':'first', 'mid_budget': 'first',
                        'big_budget': 'first', 'day': 'first', 'month': 'first', 'year': 'first', 'GENRE_ACTION': 'sum', 
                        'GENRE_ADVENTURE': 'sum', 'GENRE_ANIMATION': 'sum', 'GENRE_COMEDY': 'sum', 'GENRE_CRIME': 'sum', 
                        'GENRE_DOCUMENTARY': 'sum', 'GENRE_DRAMA': 'sum', 'GENRE_FAMILY': 'sum', 'GENRE_FANTASY': 'sum', 
                        'GENRE_FOREIGN': 'sum', 'GENRE_HISTORY': 'sum', 'GENRE_HORROR': 'sum', 'GENRE_MUSIC': 'sum', 
                        'GENRE_MYSTERY': 'sum', 'GENRE_ROMANCE': 'sum', 'GENRE_SCIENCE_FICTION': 'sum', 
                        'GENRE_THRILLER': 'sum', 'GENRE_WAR': 'sum', 'GENRE_WESTERN': 'sum', 'runtime': 'first', 
                        'popularity': 'first', 'vote_average': 'first','vote_count': 'first','flop': 'first',
                        'bomb': 'first', 'mediocre': 'first', 'success': 'first', 'blockbuster': 'first', 'super_hit': 'first'}).reset_index()

# commented out code below makes each profit class discrete, unused for the current model
'''
df_genre['mediocre'] = np.where((df_genre['super_hit'] == 1) & (df_genre['mediocre'] == 1), 0, df_genre['mediocre'])
df_genre['mediocre'] = np.where((df_genre['blockbuster'] == 1) & (df_genre['mediocre'] == 1), 0, df_genre['mediocre'])
df_genre['mediocre'] = np.where((df_genre['success'] == 1) & (df_genre['mediocre'] == 1), 0, df_genre['mediocre'])
df_genre['success'] = np.where((df_genre['super_hit'] == 1) & (df_genre['success'] == 1), 0, df_genre['success'])
df_genre['success'] = np.where((df_genre['blockbuster'] == 1) & (df_genre['success'] == 1), 0, df_genre['success'])
df_genre['blockbuster'] = np.where((df_genre['super_hit'] == 1) & (df_genre['blockbuster'] == 1), 0, df_genre['blockbuster']) 
df_genre['bomb'] = np.where((df_genre['flop'] == 1) & (df_genre['bomb'] == 1), 0, df_genre['bomb'])      
'''

# commented code for creating recent profit projections
#template_for_predictions = df_genre.head()
#template_for_predictions.to_csv('template.csv')

'''
178 - 246
select a response/dependent variable by uncommenting the dataset with it included in df_dataset instantiation
currently set to predict super hits
'''

'''
# flop dataset
df_dataset = df_genre[['id','low_budget', 'mid_budget', 'big_budget', 'day', 'month', 
                     'year', 'GENRE_ACTION', 'GENRE_ADVENTURE', 'GENRE_ANIMATION',
       'GENRE_COMEDY', 'GENRE_CRIME', 'GENRE_DOCUMENTARY', 'GENRE_DRAMA',
       'GENRE_FAMILY', 'GENRE_FANTASY', 'GENRE_FOREIGN', 'GENRE_HISTORY',
       'GENRE_HORROR', 'GENRE_MUSIC', 'GENRE_MYSTERY', 'GENRE_ROMANCE',
       'GENRE_SCIENCE_FICTION', 'GENRE_THRILLER', 'GENRE_WAR',
       'GENRE_WESTERN', 'runtime', 'popularity', 'vote_average', 'vote_count',
       'flop']]

# bomb dataset

df_dataset = df_genre[['id','low_budget', 'mid_budget', 'big_budget', 'day', 'month', 
                     'year', 'GENRE_ACTION', 'GENRE_ADVENTURE', 'GENRE_ANIMATION',
       'GENRE_COMEDY', 'GENRE_CRIME', 'GENRE_DOCUMENTARY', 'GENRE_DRAMA',
       'GENRE_FAMILY', 'GENRE_FANTASY', 'GENRE_FOREIGN', 'GENRE_HISTORY',
       'GENRE_HORROR', 'GENRE_MUSIC', 'GENRE_MYSTERY', 'GENRE_ROMANCE',
       'GENRE_SCIENCE_FICTION', 'GENRE_THRILLER', 'GENRE_WAR',
       'GENRE_WESTERN', 'runtime', 'popularity', 'vote_average', 'vote_count',
       'bomb']]


# mediocre dataset

df_dataset = df_genre[['id','low_budget', 'mid_budget', 'big_budget', 'day', 'month', 
                     'year', 'GENRE_ACTION', 'GENRE_ADVENTURE', 'GENRE_ANIMATION',
       'GENRE_COMEDY', 'GENRE_CRIME', 'GENRE_DOCUMENTARY', 'GENRE_DRAMA',
       'GENRE_FAMILY', 'GENRE_FANTASY', 'GENRE_FOREIGN', 'GENRE_HISTORY',
       'GENRE_HORROR', 'GENRE_MUSIC', 'GENRE_MYSTERY', 'GENRE_ROMANCE',
       'GENRE_SCIENCE_FICTION', 'GENRE_THRILLER', 'GENRE_WAR',
       'GENRE_WESTERN', 'runtime', 'popularity', 'vote_average', 'vote_count',
       'mediocre']]


# success dataset
'''
df_dataset = df_genre[['id','low_budget', 'mid_budget', 'big_budget', 'day', 'month', 
                     'year', 'GENRE_ACTION', 'GENRE_ADVENTURE', 'GENRE_ANIMATION',
       'GENRE_COMEDY', 'GENRE_CRIME', 'GENRE_DOCUMENTARY', 'GENRE_DRAMA',
       'GENRE_FAMILY', 'GENRE_FANTASY', 'GENRE_FOREIGN', 'GENRE_HISTORY',
       'GENRE_HORROR', 'GENRE_MUSIC', 'GENRE_MYSTERY', 'GENRE_ROMANCE',
       'GENRE_SCIENCE_FICTION', 'GENRE_THRILLER', 'GENRE_WAR',
       'GENRE_WESTERN', 'runtime', 'popularity', 'vote_average', 'vote_count',
       'success']]


# blockbuster dataset
'''
df_dataset = df_genre[['id','low_budget', 'mid_budget', 'big_budget', 'day', 'month', 
                     'year', 'GENRE_ACTION', 'GENRE_ADVENTURE', 'GENRE_ANIMATION',
       'GENRE_COMEDY', 'GENRE_CRIME', 'GENRE_DOCUMENTARY', 'GENRE_DRAMA',
       'GENRE_FAMILY', 'GENRE_FANTASY', 'GENRE_FOREIGN', 'GENRE_HISTORY',
       'GENRE_HORROR', 'GENRE_MUSIC', 'GENRE_MYSTERY', 'GENRE_ROMANCE',
       'GENRE_SCIENCE_FICTION', 'GENRE_THRILLER', 'GENRE_WAR',
       'GENRE_WESTERN', 'runtime', 'popularity', 'vote_average', 'vote_count',
       'blockbuster']]


# super_hit dataset

df_dataset = df_genre[['id','low_budget', 'mid_budget', 'big_budget', 'day', 'month', 
                     'year', 'GENRE_ACTION', 'GENRE_ADVENTURE', 'GENRE_ANIMATION',
       'GENRE_COMEDY', 'GENRE_CRIME', 'GENRE_DOCUMENTARY', 'GENRE_DRAMA',
       'GENRE_FAMILY', 'GENRE_FANTASY', 'GENRE_FOREIGN', 'GENRE_HISTORY',
       'GENRE_HORROR', 'GENRE_MUSIC', 'GENRE_MYSTERY', 'GENRE_ROMANCE',
       'GENRE_SCIENCE_FICTION', 'GENRE_THRILLER', 'GENRE_WAR',
       'GENRE_WESTERN', 'runtime', 'popularity', 'vote_average', 'vote_count',
       'super_hit']]
'''
# selects the above uncommented df_dataset as the dataset for the model
dataset = df_dataset.values
#np.random.shuffle(dataset)

# 253-261 splits training/test data
X = dataset[:,0:30]
scaler = MinMaxScaler()
normalized = scaler.fit_transform(X)
inverse = scaler.inverse_transform(normalized)
Y = dataset[:,30]
xtrain = normalized[646:]
ytrain = Y[646:]
xval = normalized[:646]
yval = Y[:646]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()
model = Sequential()
model.add(Dense(50, input_dim=30, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(50, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

#compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# fix random number seed for repeatability
seed=7
np.random.seed(seed)
# early stopping to speed up process
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
estimator = model.fit(xtrain, ytrain, batch_size=1, epochs=100, verbose=1, validation_data=(xval,yval), callbacks=[es])
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# plots the accuracy, validation accuracy & validation loss
plt.plot(estimator.history['accuracy'])
plt.plot(estimator.history['val_accuracy'])
plt.plot(estimator.history['val_loss'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy','val_accuracy','val_loss'], loc='best')
plt.show()
stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)
model.summary()

# 301-306 prints a confusion matrix of the validation data
from sklearn.metrics import confusion_matrix
predictions = model.predict_on_batch(xval)
predictions[ predictions >= .5] = 1
predictions[ predictions < .5] = 0

print(confusion_matrix(yval, predictions))

# 309-312 prints a classification report of the validation data
from sklearn.metrics import classification_report
predictions = model.predict_on_batch(xval)
predictions[ predictions >= .5] = 1
predictions[ predictions < .5] = 0

print(classification_report(yval, predictions))

# plots the model
from keras.utils import plot_model
plot_model(model, show_shapes=True, to_file='model.png')

# pulls recent film data formatted like the model data
recent_films = pd.read_csv("recent_movies.csv", delimiter = ",", header = 0)

# 324 - 335 calls the model to predict based on the recent film data
prediction_dataset = recent_films[['id','low_budget', 'mid_budget', 'big_budget', 'day', 'month', 
                     'year', 'GENRE_ACTION', 'GENRE_ADVENTURE', 'GENRE_ANIMATION',
       'GENRE_COMEDY', 'GENRE_CRIME', 'GENRE_DOCUMENTARY', 'GENRE_DRAMA',
       'GENRE_FAMILY', 'GENRE_FANTASY', 'GENRE_FOREIGN', 'GENRE_HISTORY',
       'GENRE_HORROR', 'GENRE_MUSIC', 'GENRE_MYSTERY', 'GENRE_ROMANCE',
       'GENRE_SCIENCE_FICTION', 'GENRE_THRILLER', 'GENRE_WAR',
       'GENRE_WESTERN', 'runtime', 'popularity', 'vote_average', 'vote_count']]

normed_pred_dataset = scaler.fit_transform(prediction_dataset)
predict_recents = model.predict_on_batch(normed_pred_dataset)
#predict_recents[ predict_recents >= .5] = 1
#predict_recents[ predict_recents < .5] = 0
