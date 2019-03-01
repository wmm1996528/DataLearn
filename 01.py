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
