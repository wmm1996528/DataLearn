import pandas
from sklearn.model_selection import train_test_split

train_data = pandas.read_csv('train.csv')
print(train_data.head(0))
# train_data = pandas.read_csv('train_set.csv')
test = train_data[['id', 'article', 'word_seg']]
test_target = train_data['class']
X_train, X_test, y_train, y_test = train_test_split(test, test_target, test_size=0.3, random_state=2019)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
# 第二步　计算ＴＦ
from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer()
count_vec_res = count_vec.fit_transform(X_train['word_seg'])
# 统计词频 TF
for key, value in count_vec.vocabulary_.items():
    print(key, value)

# 计算ＴＦ－ＩＤＦ
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_transformer = TfidfVectorizer()
word_seg_tfidf = tfidf_transformer.fit_transform(X_train['word_seg'])
for key, value in tfidf_transformer.vocabulary_.items():
    print(key, value)
