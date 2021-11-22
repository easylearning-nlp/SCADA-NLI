import jieba
import jieba.analyse as anls  # 关键词提取
import codecs
#1、读取文本
text = codecs.open("a.txt", 'r', encoding='utf-8').read()
#加载停用词表
stopwords = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]  # list类型
#分词未去停用词
text_split = jieba.cut(text)  # 未去掉停用词的分词结果   list类型
 
#去掉停用词的分词结果  list类型
text_split_no = []
for word in text_split:
    if word not in stopwords:
        text_split_no.append(word)
        fW = codecs.open('text_split_no.txt', 'w', encoding='UTF-8')
        fW.write(' '.join(text_split_no))

 
 