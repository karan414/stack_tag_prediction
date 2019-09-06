from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

preprocessed_data = pd.read_csv("./csv/preprocessed_questions_random_two_million_rows.csv", header=None, names=["question", "tags"])
print(preprocessed_data.head())

# binary='true' will give a binary vectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary='true')
multilabel_y = vectorizer.fit_transform(preprocessed_data['tags'].values.astype(str))


def tags_to_choose(n):
    tags = multilabel_y.sum(axis=0).tolist()[0]
    sorted_tags = sorted(range(len(tags)), key=lambda i: tags[i], reverse=True)
    multilabel_yn = multilabel_y[:, sorted_tags[:n]]
    return multilabel_yn


def questions_explained_fn(n):
    multilabel_yn = tags_to_choose(n)
    x = multilabel_yn.sum(axis=1)
    return np.count_nonzero(x == 0)


questions_explained = []
total_tags = multilabel_y.shape[1]
total_questions = preprocessed_data.shape[0]
for i in range(500, total_tags, 100):
    questions_explained.append(np.round(((total_questions-questions_explained_fn(i))/total_questions)*100,3))

fig, ax = plt.subplots()
ax.plot(questions_explained)
xlabel = list(500+np.array(range(-50, 450, 50))*50)
ax.set_xticklabels(xlabel)
plt.xlabel("Number of tags")
plt.ylabel("Number of Questions covered partially")
plt.grid()
plt.show()
print("with ", 5500, "tags we are covering ", questions_explained[50], "% of questions")

multilabel_yx = tags_to_choose(5500)
print("Number of questions that are not covered: ", questions_explained_fn(5500), "out of ", total_questions)
print("Number of tags in sample :", multilabel_y.shape[1])
print("number of tags taken: ", multilabel_yx.shape[1], "(", (multilabel_yx.shape[1]/multilabel_y.shape[1])*100, "%)")

total_size = preprocessed_data.shape[0]
train_size = int(0.80*total_size)

x_train = preprocessed_data.head(train_size)
x_test = preprocessed_data.tail(total_size - train_size)

y_train = multilabel_yx[0:train_size, :]
y_test = multilabel_yx[train_size:total_size, :]

print("Number of data points in train data: ", y_train.shape)
print("Number of data points in test data: ", y_test.shape)

vectorizer = TfidfVectorizer(min_df=0.00009, max_features=200000, smooth_idf=True, norm="l2",
                             tokenizer=lambda x: x.split(), sublinear_tf=False, ngram_range=(1, 4))
x_train_multilabel = vectorizer.fit_transform(x_train['question'])
x_test_multilabel = vectorizer.transform(x_test['question'])

print("Dimensions of train data X: ", x_train_multilabel.shape, "Y: ", y_train.shape)
print("Dimensions of test data X: ", x_test_multilabel.shape, "Y: ", y_test.shape)

classifier = OneVsRestClassifier(SGDClassifier(loss='hinge', alpha=0.00001, penalty='l1'), n_jobs=-1)
classifier.fit(x_train_multilabel, y_train)
predictions = classifier.predict(x_test_multilabel)

print("Accuracy:", metrics.accuracy_score(y_test, predictions))
print("Macro f1 score:", metrics.f1_score(y_test, predictions, average='macro'))
print("Micro f1 score:", metrics.f1_score(y_test, predictions, average='micro'))
print("Hamming loss:", metrics.hamming_loss(y_test, predictions))
print("Precision recall report: \n", metrics.classification_report(y_test, predictions))

# Dumping model
joblib.dump(classifier, './model/equal_weight_svm_model.pkl')
