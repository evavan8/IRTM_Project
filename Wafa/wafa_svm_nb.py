import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, svm
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


run_naivebayes = True
run_svm = True


# help from: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

labeled = pd.read_pickle('Manual Classification/labeled.pkl')
unlabeled = pd.read_pickle('Manual Classification/unlabeled.pkl')

df_original = pd.read_csv('Manual Classification/Wafa_classified.csv')

# Splitting classified set into training and test set
train, test = train_test_split(labeled, test_size=0.2)


# ------------------------- DEALING WITH IMBALANCED DATA --------------------------------------------------------------

""" REPORT
0	Nederzettingen en kolonistengeweld
1	Israëlische veiligheidstroepen & bezettingsautoriteiten

Problem encountered: out of 1143 unlabeled articles, 913 are classified as category 1...
Is this due to an imbalance in the data?

In training set: 72 total, 44 classified as 1 --> 61%

Naive Bayes scores with default parameters
#         Before upsampling: 84.21052631578947
#         After            : 68.42105263157895
#                            57.89473684210527
#                            63.1578947368421  --> much lower!

SVM (initially with parameters: C=1.0, kernel='linear', degree=3, gamma='auto')
#         Before upsampling: 89.47368421052632
#         After            : 84.21052631578947
#                            78.94736842105263
#                            89.47368421052632


"""
# Upsample minority class (https://elitedatascience.com/imbalanced-classes)
value_counts = train.Class.value_counts()
print(f'Value counts: \n'
      f'{value_counts}')
majority_class = value_counts.index[value_counts.argmax()]
minority_class = value_counts.index[value_counts.argmin()]
print(f'Majority class: {majority_class} \n'
      f'Minority class: {minority_class}')
train_majority = train[train.Class == majority_class]
train_minority = train[train.Class == minority_class]

num_to_add = value_counts.max() - value_counts.min()
train_minority_upsampled = train_minority.sample(n=num_to_add)

train = pd.concat([train_minority_upsampled, train_majority])


# ------------------------- TF-IDF -------------------------------------------------------------------------------

tfidf_vect = TfidfVectorizer(max_features=5000)
tfidf_vect.fit(pd.concat([train['preprocessed_body'], test['preprocessed_body'], unlabeled['preprocessed_body']]))
train_X_tfidf = tfidf_vect.transform(train['preprocessed_body'])
test_X_tfidf = tfidf_vect.transform(test['preprocessed_body'])

print(tfidf_vect.vocabulary_)
print(train_X_tfidf)

idx_to_word = tfidf_vect.get_feature_names()

""" report: tf-idf scores for some words
LOW SCORES
    0.02 --> idx_to_word[54] = 'accord'
    0.0378 --> idx_to_word[791] = 'caus'
HIGH SCORES
    0.26 --> idx_to_word[3816] = 'settler' 
    0.15 --> idx_to_word[4931] = 'yesh'

max score: 0.699
"""

# ------------------------------------- NAIVE BAYES --------------------------------------------------------------
# no grid search because naive bayes doesn't really have parameters (other than smoothing parameter)

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(train_X_tfidf, train['Class'])
# predict the labels on validation dataset
predictions_NB = Naive.predict(test_X_tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, test['Class'])*100)

# ---------------------------------------- SVM ----------------------------------------------------------

# SVM + grid search (originally: C=1.0, kernel='linear', degree=3, gamma='auto');
""" Report - parameters for SVM (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- C: float, default=1.0
      Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. 
      The penalty is a squared l2 penalty.
--> the larger C, the larger the chance of overfitting (large C = high variance, low bias)
--> large C = narrow margin, small C = soft (wide) margin

- kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
      Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, 
      ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute 
      the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).

- degree: int, default=3
      Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
--> large degree = high variance, low bias (prone to overfitting)

- gamma: {‘scale’, ‘auto’} or float, default=’scale’
      Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
      if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
      if ‘auto’, uses 1 / n_features.
      https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html:
      "Intuitively, the gamma parameter defines how far the influence of a single training example reaches, with low 
      values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as the inverse of the 
      radius of influence of samples selected by the model as support vectors."
--> too large = only support vectors have influence = overfitting
"""

parameters_svm = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
                  'degree':[2, 3, 5, 10], 'gamma':['auto', 'scale', 0.01, 0.1, 1, 10, 100]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters_svm)
clf.fit(train_X_tfidf, train['Class'])
results_df = pd.DataFrame(clf.cv_results_)
print(results_df)
#sorted(clf.cv_results_.keys())
print(f'Best estimator: {clf.best_params_} ({clf.best_score_})')


""" 
REPORT:

# Grid Search: uses the default 5-fold cross validation
trying 1120 values

Results of couple runs:

1. Best estimator: {'C': 2, 'degree': 2, 'gamma': 'scale', 'kernel': 'sigmoid'} (0.9)
2. Best estimator: {'C': 2, 'degree': 2, 'gamma': 'scale', 'kernel': 'sigmoid'} (0.9)
3. Best estimator: {'C': 2, 'degree': 2, 'gamma': 'scale', 'kernel': 'sigmoid'} (0.9)

After removing 'sigmoid':
1. Best estimator: {'C': 2, 'degree': 2, 'gamma': 'auto', 'kernel': 'linear'} (0.85)
2. Best estimator: {'C': 2, 'degree': 2, 'gamma': 'auto', 'kernel': 'linear'} (0.85)
3. Best estimator: {'C': 2, 'degree': 2, 'gamma': 'auto', 'kernel': 'linear'} (0.85)
"""


# initialize svm with best params:
SVM = svm.SVC(C=clf.best_params_['C'], kernel=clf.best_params_['kernel'], degree=clf.best_params_['degree'], gamma=clf.best_params_['gamma'])
SVM.fit(train_X_tfidf, train['Class'])
# predict the labels on validation dataset
predictions_SVM_test = SVM.predict(test_X_tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM_test, test['Class'])*100)       # 94.73684210526315
print(f'Confusion matrix: \n'
      f'{confusion_matrix(test.Class, predictions_SVM_test)}')
#   [TP  FP
#    FN  TN]
plot_confusion_matrix(SVM, test_X_tfidf, test.Class)
plt.show()

""" 

                    Predicted label
                    0.0             1.0
            0.0     TP              FP
True label  
            1.0     FN              TN


Best estimator: {'C': 2, 'degree': 2, 'gamma': 'auto', 'kernel': 'linear'} (0.9217948717948719)
    SVM Accuracy Score ->  84.21052631578947
    Confusion matrix: 
        [[ 4  3]
         [ 0 12]]

         
"""

# predict the unlabeled data (with SVM: highest scores)
unlabeled_X_tfidf = tfidf_vect.transform(unlabeled['preprocessed_body'])
predictions_SVM = SVM.predict(unlabeled_X_tfidf)
unlabeled['Class'] = predictions_SVM

# output predicted (training+test included) dataset
df_original.loc[unlabeled.index, 'Class'] = predictions_SVM
df_original.to_csv('SVM_NB/original_plus_predicted_SVM.csv', index=False)
unlabeled.to_csv('SVM_NB/predicted_only_SVM.csv')


