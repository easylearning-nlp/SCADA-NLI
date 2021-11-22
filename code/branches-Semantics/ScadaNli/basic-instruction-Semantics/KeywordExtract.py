
import jieba.analyse
from jieba import lcut
from jieba import posseg
from time_recognition import time_extract
from init import my_initialize


stopword_file = './src/stopwords.txt'

# 载入字典和停用词表
def creat_stopword_list():
    stopwords = []
    for word in open(stopword_file, 'r', encoding='utf-8'):
        stopwords.append(word.strip())
    return  stopwords

# 删除停用词
def del_stopwords(sentence):
    words = lcut(sentence, cut_all = False)
    # print(words)
    stopwords = creat_stopword_list()
    stayed_words = ""
    for word in words:
        if word not in stopwords:
            stayed_words += word
    # print(stayed_words)
    return stayed_words

# 关键词提取
def keyword_extract(sentence, algorithm):
    res = []
    time = time_extract(sentence)
    # TF-TDF 算法
    if algorithm == 'TF':
        keywords = jieba.analyse.extract_tags(del_stopwords(sentence), topK=5, withWeight=True, allowPOS = (
            'ns', 'n', 'vn', 'v', 'nz', 'nc'))
    # TextRank算法
    elif algorithm == "TR":
        keywords = jieba.analyse.textrank(del_stopwords(sentence), topK=5, withWeight=True,allowPOS=(
            'ns', 'n', 'vn', 'v', 'nz', 'nc'))
    else:
        print("Error")
        exit(0)
    # print(keywords)
    for item in keywords:
        res.append(item[:][0])
    ls = [time, res]
    return ls

'''
Tag
输入：[词1, 词2, ...]
返回：[[词1，词性], [词2，词性], ... ]
'''
# def word_tag(sentence):
#     ls = []
#     words = jieba.posseg.cut(sentence)
#     print(words)
#     for word in words:
#         ls.append([word.word,word.flag])
#     return ls

def word_tag(list):
    ls = []
    for i in list:
        words = jieba.posseg.cut(i)
        for word in words:
            ls.append([word.word, word.flag])
    return ls


if __name__ == '__main__':
    my_initialize()
    text = '帮我打开卧室的空调'
    sentence = keyword_extract(del_stopwords(text), 'TF')
    print(sentence)
    print(word_tag(sentence[1]))
    # print(word_tag(ls))
