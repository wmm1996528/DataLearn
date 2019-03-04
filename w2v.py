import pandas as pd
import gensim
import time
import pickle
import numpy as np
import csv, sys

vector_size = 100

maxInt = sys.maxsize
decrement = True
while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt / 10)
        decrement = True


# =======================================================================================================================
# 0 辅助函数
# =======================================================================================================================

def sentence2list(sentence):
    return sentence.strip().split()


start_time = time.time()

data_path = '../'
feature_path = './'
proba_path = './'
model_path = './'
result_path = './'
# =======================================================================================================================
# 1 准备训练数据
# =======================================================================================================================

print("准备数据................ ")
df_train = pd.read_csv(data_path + 'train_set1.csv', engine='python')
df_test = pd.read_csv(data_path + 'test_set1.csv', engine='python')
sentences_train = list(df_train.loc[:, 'word_seg'].apply(sentence2list))
sentences_test = list(df_test.loc[:, 'word_seg'].apply(sentence2list))
sentences = sentences_train + sentences_test
print("准备数据完成! ")

# =======================================================================================================================
# 2 训练
# =======================================================================================================================
print("开始训练................ ")
model = gensim.models.Word2Vec(sentences=sentences, size=vector_size, window=5, min_count=5, workers=8, sg=0, iter=5)
print("训练完成! ")

# =======================================================================================================================
# 3 提取词汇表及vectors,并保存
# =======================================================================================================================
print(" 保存训练结果........... ")
wv = model.wv
vocab_list = wv.index2word
word_idx_dict = {}
for idx, word in enumerate(vocab_list):
    word_idx_dict[word] = idx

vectors_arr = wv.vectors
vectors_arr = np.concatenate((np.zeros(vector_size)[np.newaxis, :], vectors_arr), axis=0)  # 第0位置的vector为'unk'的vector

f_wordidx = open(feature_path + 'word_seg_word_idx_dict.pkl', 'wb')
f_vectors = open(feature_path + 'word_seg_vectors_arr.pkl', 'wb')
pickle.dump(word_idx_dict, f_wordidx)
pickle.dump(vectors_arr, f_vectors)
f_wordidx.close()
f_vectors.close()
print("训练结果已保存到该目录下！ ")

end_time = time.time()
print("耗时：{}s ".format(end_time - start_time))