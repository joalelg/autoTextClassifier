#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:54:56 2019

@author: josalop
"""

from sklearn import (model_selection, preprocessing, linear_model, naive_bayes
                     , metrics, svm)
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn import decomposition, ensemble

import xgboost, textblob
import string
import numpy as np
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers


# Pandas display options 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

container_path = "input/"
cat_names = list(os.walk(container_path))[0][1]
documents = load_files(container_path, categories = cat_names
                       , encoding='utf-8', decode_error='ignore')
dir(documents) #['DESCR', 'data', 'filenames', 'target', 'target_names']


# Split on train and test sets
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(documents.data
                                      , documents.target)

print('Train and test documents: %d, %d'% (len(X_train), len(X_valid)))
#==========================Count Vectors as features ==========================
# 1.1  Extract vocabulary, Create count Vectors
# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(documents.data)

# 1.2 . b Tag vocabulary

# We build a (word, POS) dictionary access vocabulary by POS, p.ex. selecting nouns
# containing tuples (POS_TAG, count_vectorizer_word_index)
pos_vocabulary = {}
for word in list(count_vect.vocabulary_.keys()):
    word_pos = textblob.TextBlob(word).tags   #  → [('word','pos_tag')]
    pos_vocabulary[word] = word_pos[0][1]
    
for word in list(count_vect.vocabulary_.keys())[0:10]:
    print(word, pos_vocabulary[word])


#np.unique([pos_vocabulary[w] for w in pos_vocabulary.keys()])
#array(['CC', 'CD', 'DT', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
#       'NNS', 'PRP', 'PRP$', 'RB', 'RBR', 'TO', 'VB', 'VBD', 'VBG', 'VBN',
#       'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'], dtype='<U4'
# =======================Feature Engineering =================================
# 2.1 Count Vectors as features
# transform the training and validation data using count vectorizer object
X_train_count =  count_vect.transform(X_train)
X_valid_count =  count_vect.transform(X_valid)


# 2.2 TF-IDF Vectors as features
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
tfidf_vect.fit(X_train)
X_train_tfidf =  tfidf_vect.transform(X_train)
X_valid_tfidf =  tfidf_vect.transform(X_valid)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}'
                                   , ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit_transform(X_train)
X_train_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
X_valid_tfidf_ngram =  tfidf_vect_ngram.transform(X_valid)



## 2.3 Word Embeddings
#'''
#Fn to get dictionary of embeddings values are n_dim embeddings
#'''
#def load_vectors(fname):
#    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#    n, d = map(int, fin.readline().split())
#    data = {}
#    for line in fin:
#        tokens = line.rstrip().split(' ')
#        data[tokens[0]] = map(float, tokens[1:])
#    return data
#
#embeddings_index = load_vectors('data/wiki-news-300d-1M.vec')

# 2.4 Topic Models

# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(
                          n_components=len(cat_names)*4, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(X_train_count)
topic_words = lda_model.components_  #shape (d_topics, n_words)
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 20
pos_subset_dic = {'NN': 'Noun', 'NNP':'ProperNoun', 'VB':'Verb', 'JJ':'Adjective' }

topic_summaries = []
pos_filter = False #POS Filter to p.ex. only NN and VB
for i, topic_dist in enumerate(topic_words):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    if pos_filter == True:
        filtered_topic_words =  [w  for w in  topic_words if  pos_vocabulary[w] in pos_subset_dic.keys()]
        top__topic_words = filtered_topic_words
    top__topic_words = topic_words
    topic_summaries.append(' '.join(top__topic_words))
    print("Topic %d: %s" % (i,topic_summaries[i]))
    
    
# =======================Building the Models=================================
    
performance_metric = 'accuracy_score'
def train_model(model, feature_vect_train, target_train, feature_vect_valid
                , metric = 'accuracy_score'):
    ''' 
        Fit the training dataset on the model  
    '''
    model.fit(feature_vect_train, target_train)
    
    # predict the labels on validation dataset
    predictions = model.predict(feature_vect_valid)
    
    return getattr(metrics, metric)(predictions, y_valid)


# Explore categorie distributios
category_proportions = pd.Series(pd.Series(documents.target).value_counts() / len(
        documents.target)*100, name = 'ptj').sort_index().round(2)
category_proportions

# Set base model classifying to bigger category, i.e. classif all to tags median
predictions = np.ones(shape = (y_valid.shape)) * category_proportions.values.argmax()

base_performance = getattr(metrics, performance_metric)(predictions, y_valid)
print("Base model tags median as prediction %s: %.4f"% (performance_metric, base_performance))

# Naive Bayes on Count Vectors
performance = train_model(naive_bayes.MultinomialNB(), X_train_count, y_train
                          , X_valid_count, metric=performance_metric)
print("NB, Count Vectors: %s: %.4f" % (performance_metric, performance) )

# Naive Bayes on Word Level TF IDF Vectors
performance = train_model(naive_bayes.MultinomialNB(), X_train_tfidf, y_train
                       , X_valid_tfidf, metric=performance_metric)
print("NB, WordLevel TF-IDF: %s %.4f" % (performance_metric, performance) )

# Naive Bayes on Ngram Level TF IDF Vectors
performance = train_model(naive_bayes.MultinomialNB(), X_train_tfidf_ngram, y_train,
                       X_valid_tfidf_ngram, metric=performance_metric)
print("NB, N-Gram Vectors: %s %.4f" % (performance_metric, performance) )


# Linear Classifier on Count Vectors
performance = train_model(linear_model.LogisticRegression(), X_train_count, y_train
                       , X_valid_count, performance_metric)
print("LR, Count Vectors: %s %.4f" % (performance_metric, performance) )

# Linear Classifier on Word Level TF IDF Vectors
performance = train_model(linear_model.LogisticRegression(), X_train_tfidf, y_train
                       , X_valid_tfidf, performance_metric)
print("LR, WordLevel TF-IDF: %s %.4f" % (performance_metric, performance) )

# Linear Classifier on Ngram Level TF IDF Vectors
performance = train_model(linear_model.LogisticRegression(), X_train_tfidf_ngram, y_train
                       , X_valid_tfidf_ngram, performance_metric)
print("LR, N-Gram Vectors: %s %.4f" % (performance_metric, performance) )

# SVM on Ngram Level TF IDF Vectors
performance = train_model(svm.SVC(), X_train_tfidf_ngram, y_train, X_valid_tfidf_ngram)
print("SVM, N-Gram Vectors: %s %.4f" % (performance_metric, performance) )


# Bagging tree models

# RF on Count Vectors
performance = train_model(ensemble.RandomForestClassifier(max_features= 'auto'), X_train_count, y_train
                       , X_valid_count, performance_metric)
print("RF, Count Vectors: %s %.4f" % (performance_metric, performance) )

# RF on Word Level TF IDF Vectors
performance = train_model(ensemble.RandomForestClassifier(max_features = 'auto'), X_train_tfidf, y_train
                       , X_valid_tfidf, performance_metric)
print("RF, WordLevel TF-IDF: %s %.4f" % (performance_metric, performance) )



# Extereme Gradient Boosting on Count Vectors
performance = train_model(xgboost.XGBClassifier(), X_train_count.tocsc(), y_train
                       , X_valid_count.tocsc(), performance_metric)
print("Xgb, Count Vectors: %s %.4f" % (performance_metric, performance) )

# Extereme Gradient Boosting on Word Level TF IDF Vectors
performance = train_model(xgboost.XGBClassifier(), X_train_tfidf.tocsc(), y_train,
                       X_valid_tfidf.tocsc(), performance_metric)
print("Xgb, WordLevel TF-IDF: %s %.4f" % (performance_metric, performance) )

    
# =======================Training Pipeline=================================
from sklearn import ( feature_selection, pipeline, #cross_validation, grid_search
                     preprocessing, decomposition)
from sklearn.linear_model import LogisticRegressionCV
from sklearn import model_selection
from sklearn.feature_selection import chi2
from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
from sklearn.feature_selection import SelectFromModel


# Set temp storage to cache first pipe transformations
cachedir = mkdtemp()
memory = Memory(location=cachedir, verbose=10)


# Set model for feature selection
feature_selector_model  = LogisticRegressionCV(Cs=10, n_jobs=2, penalty='l2')
cached_pipe = pipeline.Pipeline([
          ('vectorizer', CountVectorizer(analyzer='word', token_pattern=r'\w{1,}'))
         ,('tfidf', TfidfTransformer())
         ,('reduce_dim', decomposition.TruncatedSVD())
         ,('scaler', preprocessing.MaxAbsScaler())
         ,('feature_select',  SelectFromModel(feature_selector_model))
         ] #, ('top_ftrs', top_ftr_selector)
         , memory=memory
         )

params_grid = {
                'tfidf__use_idf': (True, False) #
               , 'vectorizer__max_features': [5000] #, 5000, 10000, 50000
               , 'vectorizer__max_df': [0.5] #, 0.75, 1.0
               , 'vectorizer__ngram_range': [(1, 1)]  #, (1, 2) unigrams or bigrams
               , 'reduce_dim__n_components': [300] #50, 100, 200 ,
               , 'feature_select__threshold': [.3] #.1, .2,
             }


# Set models and hyperparameters
models = [
          # ('nb', naive_bayes.MultinomialNB())
          ('lasso', linear_model.LogisticRegression(penalty='l2'))
          , ('svm', svm.SVC())
          , ('rf', ensemble.RandomForestClassifier(n_estimators = 200))
          , ('xgb', xgboost.XGBClassifier(n_estimators = 200))
          ]

modelsDic = dict(models)


model_params = {
              #  'lasso__C': [.001,.01, .1, 1, 10, 100, 1000, 10000, 100000]
              #, 'svm__C': [.001,.01, .1, 1, 10, 100, 1000, 10000, 100000]
              #, 'svm__kernel':  ['linear', 'rbf']
              #, 'rf__max_features':['sqrt', 'log2']
              #, 'xgb__max_depth':[3]
              #, 'xgb__colsample_bytree':[.1, .2, .4]
              #, 'xgb__subsample': [.8]
              #, 'xgb__gamma':[1]
        }

scores = {}
#ftrs_reduced_n = 300
#total_ftrs = min(X_train_count.shape[1], ftrs_reduced_n)
ftr_compress = False
folds = 5


for model_name in modelsDic.keys():
    pipe = cached_pipe
    pipe.steps.append((model_name, modelsDic[model_name]))
    
    #Add model parameters to the grid dict    
    for key in model_params.keys():
        if model_name in key:
            params_grid[key]:model_params[key]

    gs = model_selection.GridSearchCV(estimator=pipe, param_grid = params_grid 
                                  , cv = folds, n_jobs=1,  scoring = 'accuracy')
    gs.fit(X_train, y_train)
    scores[model_name] = None
    scores[model_name]={'best_score':  gs.best_score_}
    print("Best {}: {:.4f} with params: {}: ".format( performance_metric
                                          , gs.best_score_, gs.best_params_))
    pipe.steps.pop()   #Pop model in turn from pipe
    if (model_name in params_grid.keys()): params_grid.pop(model_name) #Pop model in turn hyperparams from pipe
    
    # Store best model results to plot i.e  those corresponding to best model params
    cv_results_df = pd.DataFrame(gs.cv_results_)
    scores[model_name]['cv_optim_result'] = cv_results_df.loc[
                                              cv_results_df['params']==gs.best_params_,:]
    scores[model_name]['cv_optim_result'] = scores[model_name]['cv_optim_result']
    scores[model_name]['best_estimator'] = gs.best_estimator_


# Build Optimal results DF to plot
optim_results_df = pd.DataFrame(None
                        , columns = [c for c in scores[list(scores.keys())[0]]['cv_optim_result'].columns
                                                       if 'param' not in c]
                        , index = list(scores.keys()))
for m in optim_results_df.index:
    no_parameter_cols = [c for c in scores[m]['cv_optim_result'].columns if 'param' not in c]
    optim_results_df.loc[m,:] = scores[m]['cv_optim_result'].loc[:,no_parameter_cols].values


# Plot
f, ax= plt.subplots()
x,y,e = (list(optim_results_df.index), optim_results_df['mean_test_score']
         , optim_results_df['std_test_score'])
plt.errorbar(x, y, e, linestyle='None', marker='^')
plt.title('Models %d folds CV performance' % folds)
plt.ylabel('%s' % performance_metric)
plt.show()

# Delete the temporary cache before exiting
rmtree(cachedir)

from sklearn.metrics import confusion_matrix
max_overall_performance = optim_results_df.loc[:,'mean_test_score'].agg(np.max) 
best_model_name = list(optim_results_df.loc[optim_results_df['mean_test_score']
                                ==max_overall_performance,:].index)[0]
best_model = scores[best_model_name]['best_estimator']
y_valid_preds = best_model.predict(X_valid)

print('Best %s %s parameters:' % (best_model_name, scores[best_model_name]['best_estimator']))

# Confussion matrix
print('Best %s %s confussion matrix: ' % (scores[best_model_name]['best_estimator']
                                          , best_model_name))
confusion_matrix(y_valid, y_valid_preds)

# Save scroes object on disk
import pickle 

scores_pickle = open('data/scores.pkl', 'wb') if os.path.exists('data/scores.pkl') else None 
cv_results_df_pickle = open('data/cv_results_df.pkl', 'wb') if (
                                         os.path.exists('data/cv_results_df.pkl')) else None
pickle.dump(scores, scores_pickle)

# Read back
filehandler = open('data/scores.pkl', 'rb') 
scores = pickle.load(filehandler)


#====================== Results Visualization ============================
# Reduce dimension to 2 for Visualization
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
# Reduce dimension to 2 with LinearDiscriminantAnalysis
named_steps = scores[best_model_name]['best_estimator'].named_steps
preproc_named_steps = {i:named_steps[i] for i in named_steps if i!=best_model_name}
optim_preproc_pipe = pipeline.Pipeline( [(k,preproc_named_steps[k]) for k in 
                                   preproc_named_steps.keys()])


# Use optimal model  neighbor classifier to evaluate the methods
best_estimator = scores[best_model_name]['best_estimator']


n_neighbors = 3
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# List methods to be compared
dim_reduction_methods = [('SVD', 'svd2D'), ('LDA', 'lda2d'), ('NCA', 'nca2D')]
tags_codes = {}
for (i,name) in enumerate(documents.target_names):
    tags_codes[i] = name


import seaborn as sns
for i, (display_name, pipe_name) in enumerate(dim_reduction_methods):
    
    # Make pipe with all preprocess steps and 2D visualization model from optimal estimator
    named_steps = scores[best_model_name]['best_estimator'].named_steps
    preproc_named_steps = {i:named_steps[i] for i in named_steps if i!=best_model_name}
    optim_preproc_pipe = pipeline.Pipeline( [(k,preproc_named_steps[k]) for k in 
                                       preproc_named_steps.keys()])
    
    optim_preproc_pipe.steps.append((pipe_name
                                            , LinearDiscriminantAnalysis(n_components=2)))


    plt.figure()
    # plt.subplot(1, 3, i + 1, aspect=1)

    # Fit the method's model
    optim_preproc_pipe.fit(X_train, y_train)

    # Fit a nearest neighbor classifier on the embedded training set
    knn.fit(optim_preproc_pipe.transform(X_train), y_train)

    # Compute the nearest neighbor accuracy on the embedded test set
    acc_knn = knn.score(optim_preproc_pipe.transform(X_valid), y_valid)

    # Embed the data set in 2 dimensions using the fitted model
    X_embedded = optim_preproc_pipe.transform(documents.data)

    # Plot the projected points and show the evaluation score
    display_df = pd.DataFrame({
            'Dim1':X_embedded[:, 0], 'Dim2':X_embedded[:, 1]
            , 'Category': list(map(lambda x: tags_codes[x], documents.target))
            , 'Prediction': list(map(lambda x: tags_codes[x]
                                 , best_estimator.predict(documents.data)))
            })
    
    #plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=documents.target
    #            , s=30, cmap='Set1', alpha = 0.3)
    g = sns.scatterplot(x="Dim1", y="Dim2", hue="Category", data=display_df)
    plt.title("{}, Best model {} \nTest accuracy = {:.2f}".format(display_name,
                                                              best_model_name,
                                                              acc_knn))
plt.show()




# Plotly interactive plot
from plotly.offline import plot
import plotly.graph_objs as go
def text_split(txt_line, lines2show = 8):
    '''
    Function to add html separator '<br>' every 20 words to allow easy reading of long text
    '''
    tokens = txt_line.split()
    c = 0
    out_txt = ''
    # Extract first lines to display
    while c < min(20*lines2show, len(tokens)):
        out_txt = out_txt+ ' '.join(tokens[c:c+20])+'<br>'
        c += 20
    return out_txt


test_txt = 'England victory tainted by history\n\nAs England attempt to secure a series victory in South Africa, they will do so at the venue for a previous match which became the stuff of Test cricket folklore.\n\nSouth Africa\'s meeting with England at Centurion Park in January 2000 was thought to have been an enthralling spectacle, with the tourists claiming a remarkable win after three whole days were lost to bad weather. It took a few short months to reveal the unfortunate truth; that a bookmaker had given home skipper Hansie Cronje money - and a leather jacket - to influence the outcome of the match. Cronje, who was killed two years later in a plane crash, was subsequently found guilty of match-fixing and banned from Test cricket for life. Opening the bowling for England was Andy Caddick, who told BBC Sport: "They were 2-0 up in the series, we had a lot of English supporters there and South Africa just wanted to make a game of it. That\'s what I thought. "At the time you don\'t think anything of it but then afterwards you start to put two and two together, with events that happened afterwards." England captain Nasser Hussain put the South Africans in after winning the toss, and was given an early reward, of a purely cricketing nature, when Gary Kirsten edged Darren Gough to second slip for a duck in the first over.\n\nCronje also fell to Gough for a duck as the home side closed the first day 155-6. The next three days were then wiped out competely due to a combination of rain and damp ground conditions before Cronje approached Hussain with a suggestion to manufacture a result. South Africa had already secured the series having won in Johannesburg and Cape Town, but Cronje (and his friends) wanted to create some interest in the final day at Centurion. The idea was for each side to forfeit an innings each, leaving the tourists with a run chase, so South Africa reached 248-8 on the final morning to set England 249 from 76 overs. Kirsten was one of several players opposed to the move, as he explained to BBC Sport. "Hansie came up to us in the changing room on the last day and said: \'I\'m keen to forfeit the innings and set the total.\'\n\n"He asked the team and there was a mixed reception. "Some of the guys felt we didn\'t want to give England a chance of winning - you don\'t mess around with Test matches - and some said nothing! "Hansie just decided, and maybe we know why in hindsight, to go for it but there was quite a bit of resistance and some of the guys didn\'t think it was a good idea at all." England\'s 12th man that day was Phil Tufnell, who recalled: "Everyone thought it was a very good gesture in the spirit of the game. "The game had meandered along and all of a sudden it was like an old fashioned three-day county game and we were having a chase." Whatever the reasons - later it emerged that Cronje\'s part had been simply to ensure against a draw - it led to a fascinating cricket match.\n\nEngland needed six from the final seven balls and Yorkshire tailenders Chris Silverwood and Darren Gough both hit boundaries to give them a two-wicket victory. "It ended as a great day for cricket," Kirsten conceded. "One could argue it was good for people who had been frustrated having not watched any cricket and suddenly got a great last day. "But from a player\'s point of view you\'ve worked so hard for three months trying to win a Test series that you don\'t give Test matches away, or create the opportunity to make it easy to win one." Tufnell has favourable memories of Cronje, who played county cricket for Leicestershire in the 1990s. "He always came across as a nice bloke," the spinner said. "He always came and said \'Hi\' and seemed a very intelligent captain, a good competitor and a good ambassador for South African cricket." Kirsten too recalls some positive images of Cronje. "We all know that what he did cannot be condoned in any circumstances," he said. "But my view is he did a huge amount for my career. "I thought he was an outstanding captain and I\'ve made mistakes in my life so I\'m not in a position to be judgemental."\n'
text_split(test_txt)        

display_df['txt'] = pd.Series(documents.data).apply(lambda x: text_split(x))
display_df['txt'] = 'Predicted: ' + display_df.Prediction + '<br>' + display_df['txt'] 

fig = {
    'data': [
        
  		{
  			'x': display_df.loc[display_df.Category==c,'Dim1'], 
        	'y': display_df.loc[display_df.Category==c,'Dim2'],
        	'text': display_df.loc[display_df.Category==c,'txt'], 
        	'mode': 'markers', 
        	'name': c}
    for c in cat_names ],
    'layout': {
        'xaxis': {'title': 'Dim1'},
        'yaxis': {'title': "Dim2"}
    }
}

# IPython notebook
# py.iplot(fig, filename='pandas/multiple-scatter')
# Using URL
#url = py.plot(fig, filename='pandas/multiple-scatter')

plot(fig, auto_open=True)
