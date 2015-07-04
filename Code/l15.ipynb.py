
import pickle
dataset = pickle.load(open('dataset.pkl', 'rb'))


# In[29]:

from collections import Counter


# ** 2.) Filter dataset. **

# In[54]:

# Remove dataset that don't have a description.
filtered_dataset = [t for t in dataset if t['description']]
print '%d / %d dataset have description' % (len(filtered_dataset), len(dataset))





# In[185]:

# Sidebar: difference between lists and generators (iterators)

x = [1,2,3,4,5,6,7,8,9]

sublist = [i for i in x if i % 2 == 0]

sublist2 = (i for i in x if i % 2 == 0)



# **3.) Tokenize dataset. **

# In[153]:

# This time, we'll use sklearn's built in tokenization.
from sklearn.feature_extraction.text import TfidfVectorizer
def tokenize(dataset, vectorizer):
    X = vectorizer.fit_transform(t['description'] for t in dataset)
    print 'X dimensions=', X.shape
    return X

X = tokenize(filtered_dataset, TfidfVectorizer())


# ** 4.) Compute cross validation accuracy. **

# In[154]:

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

def do_cv(X, y, nfolds=10):
    cv = KFold(len(y), nfolds)
    return np.mean(cross_val_score(LogisticRegression(), X, y, cv=cv))
  
print do_cv(X, y)


# ** 5.) Compare different preprocessing decisions. **

# In[155]:

print do_cv(tokenize(filtered_dataset, TfidfVectorizer()), y)
print do_cv(tokenize(filtered_dataset, TfidfVectorizer(min_df=10)), y)


# In[156]:

# How does filtering rare words affect accuracy?
import matplotlib.pyplot as plt

def compare_mindf(dataset, y):
    accuracies = []
    for freq in range(20):
        accuracies.append(do_cv(tokenize(dataset, TfidfVectorizer(min_df=freq)), y))
    plt.figure()
    plt.plot(accuracies, 'bo-')
    plt.xlabel('mindf')
    plt.ylabel('accuracy')
    plt.show()
        
compare_mindf(filtered_dataset, y)


# In[158]:

# How does filtering common words affect accuracy?
import matplotlib.pyplot as plt

def compare_maxdf(dataset, y):
    accuracies = []
    maxdfs = [1., .1, .05, .04, .03, .02, .01, .005, .001]
    for freq in maxdfs:
        print freq
        accuracies.append(do_cv(tokenize(dataset, TfidfVectorizer(max_df=freq)), y))
    plt.figure()
    plt.plot(maxdfs, accuracies, 'bo-')
    plt.xlabel('maxdf (fraction)')
    plt.ylabel('accuracy')
    plt.show()
        
compare_maxdf(filtered_dataset, y)


# In[161]:

# How does using ngrams help?
import matplotlib.pyplot as plt

def compare_ngrams(dataset, y):
    accuracies = []
    ngrams = [(1,1), (1,2), (2,2), (1,3), (2, 3), (3,3)]
    for ngram in ngrams:
        print ngram
        accuracies.append(do_cv(tokenize(dataset, TfidfVectorizer(min_df=2, ngram_range=ngram)), y))
    plt.figure()
    plt.plot(accuracies, 'bo-')
    plt.xticks(range(len(ngrams)), ngrams)
    plt.xlabel('ngrams')
    plt.ylabel('accuracy')
    plt.show()
        
compare_ngrams(filtered_dataset, y)


# In[162]:

# Does binary/freq help?
def compare_binary(dataset, y):
    accuracies = []
    choices = [True, False]
    for choice in choices:
        print choice
        accuracies.append(do_cv(tokenize(dataset, TfidfVectorizer(binary=choice, min_df=2, ngram_range=(1,3))), y))
    plt.figure()
    plt.plot(accuracies, 'bo-')
    plt.xticks([0,1], [str(c) for c in choices])
    plt.xlabel('binary')
    plt.ylabel('accuracy')
    plt.show()
        
compare_binary(filtered_dataset, y)


# In[163]:

# Does idf help?
def compare_idf(dataset, y):
    accuracies = []
    choices = [True, False]
    for choice in choices:
        print choice
        accuracies.append(do_cv(tokenize(dataset, TfidfVectorizer(use_idf=choice, binary=True, min_df=2, ngram_range=(1,3))), y))
    plt.figure()
    plt.plot(accuracies, 'bo-')
    plt.xticks([0,1], [str(c) for c in choices])
    plt.xlabel('binary')
    plt.ylabel('use idf?')
    plt.show()
        
compare_idf(filtered_dataset, y)


# In[164]:

# Do stop words help?
def compare_stopwords(dataset, y):
    accuracies = []
    choices = [None, 'english']
    for choice in choices:
        print choice
        accuracies.append(do_cv(tokenize(dataset, TfidfVectorizer(stop_words=choice,
                                                                 use_idf=True, binary=True, min_df=2, ngram_range=(1,3))), y))
    plt.figure()
    plt.plot(accuracies, 'bo-')
    plt.xticks([0,1], [str(c) for c in choices])
    plt.xlabel('binary')
    plt.ylabel('use stopwords')
    plt.show()
        
compare_stopwords(filtered_dataset, y)


# In[125]:

# Do GridSearch on all parameters.
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from pprint import pprint

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', LogisticRegression()),
])

parameters = {
    'vect__min_df': (1, 2, 3, 4, 5),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams or trigrams
    'vect__use_idf': (True, False),
}

data = [t['user']['description'] for t in filtered_dataset]
grid_search = GridSearchCV(pipeline, parameters, verbose=1, cv=KFold(len(y), 10))

print "Performing grid search..."
print "pipeline:", [name for name, _ in pipeline.steps]
print "parameters:"
pprint(parameters)
grid_search.fit(data, y)
print "done."

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[130]:

print '\n'.join(str(x) for x in grid_search.grid_scores_)


# In[175]:

# What are top terms?
def print_top_terms(C, vectorizer, dataset, y, n=30):
    X = tokenize(dataset, vectorizer)
    clf = LogisticRegression(C=C)
    clf.fit(X, y)
    coef = clf.coef_[0]
    srted = np.argsort(coef)
    feats = vectorizer.get_feature_names()
    print '\nMale terms:\n' + '\n'.join('%s=%g' % (feats[i], coef[i]) for i in srted[:n])
    print '\nFemale terms:\n' + '\n'.join('%s=%g' % (feats[i], coef[i]) for i in srted[::-1][:n])
    
print_top_terms(1, TfidfVectorizer(min_df=2, ngram_range=(1,3)), filtered_dataset, y)


# In[172]:

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

pipeline = Pipeline([
    ('vect', TfidfVectorizer(min_df=2, ngram_range=(1,3))),
    ('fsel', SelectKBest(chi2)),
    ('clf', LogisticRegression(C=1000)),
])

parameters = {
    'fsel__k': (1000, 10000, 20000, 'all'),
}

data = [t['user']['description'] for t in filtered_dataset]
grid_search = GridSearchCV(pipeline, parameters, verbose=1, cv=KFold(len(y), 10))
grid_search.fit(data, y)
print "done."

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[173]:

print '\n'.join(str(x) for x in grid_search.grid_scores_)


