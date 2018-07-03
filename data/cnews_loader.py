# coding: utf-8

import sys
import re
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
import tensorflow as tf

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels


def read_en_file(filename):
    """
    读取英文文件
    """
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                if line != '':
                    label, title, content = line.split(',')
                    content = clean_str(content)
                    contents.append(native_content(content))
                    labels.append(int(label) - 1)
            except Exception as e:
                pass
    # print(len(contents), len(labels))
    # print(labels)
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=10000):
    """根据训练集构建词汇表，存储"""
    print("Build vocabulary")
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    print(len(counter))
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def build_en_vocab(train_dir, vocab_dir, vocab_size=10000):
    """
    英文词汇字典
    """
    print("Build en vocabulary")
    data_train, _ = read_en_file(train_dir)
    # print(data_train)

    all_data = []
    for content in data_train:
        for word in content.split(' '):
            # print(word)
            try:
                all_data.append(word)
            except Exception as e:
                pass
    
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
   #               '其他', '其他1', '其他2', '其他3', '其他4', '其他5', '其他6', '其他7', '其他8', '其他9']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def read_en_category(class_dir):
    categories = []
    with open_file(class_dir) as f:
        for line in f:
            # print(line.strip().lower())
            categories.append(line.strip().lower())
        categories = [native_content(x) for x in categories]
        cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    #按行读，加lable
    contents, labels = read_file(filename)
    data_id, label_id = [], []

    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = tf.keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = tf.keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    return x_pad, y_pad


def process_en_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    #按行读，加lable
    contents, label_id = read_en_file(filename)
    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        # label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = tf.keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = tf.keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


# build_en_vocab('dbpedia_csv/train.csv', 'vocab_en.txt', vocab_size=20000)
# read_en_category('dbpedia_csv/classes.txt')
# contents, labels = read_en_file('dbpedia_csv/train.csv')
# print(labels)