import pandas
from sklearn.model_selection import train_test_split

train_data = pandas.read_csv('../train.csv')
print(train_data.head(0))
# train_data = pandas.read_csv('train_set.csv')
test = train_data[['id', 'article', 'word_seg']]
test_target = train_data['class']
X_train, X_test, y_train, y_test = train_test_split(test, test_target, test_size=0.3, random_state=2019)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# 第二步　计算ＴＦ
# 统计词频 TF
from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer()
count_vec_res = count_vec.fit_transform(X_train['word_seg'])

# for key, value in count_vec.vocabulary_.items():
#     print(key, value)

# 计算ＴＦ－ＩＤＦ
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_transformer = TfidfVectorizer()
tfidf_transformer.fit(X_train['word_seg'])
x_train = tfidf_transformer.transform(X_train['word_seg'])
x_test = tfidf_transformer.transform(X_test['word_seg'])
# y_test = tfidf_transformer.transform(y_test['word_seg'])
# for key, value in tfidf_transformer.vocabulary_.items():
#     print(key, value)


# Word2Vec
from gensim.models import Word2Vec
model = Word2Vec(X_train['word_seg'], workers=5)#　设置工作线程数
model.save("model.pkl")

from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(y_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
print(acc_svc)
#
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
print(acc_log)