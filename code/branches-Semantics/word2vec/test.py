# coding=utf-8
import gensim

if __name__ == '__main__':
    model = gensim.models.Word2Vec.load('zhiwiki_news.word2vec')

   	# 计算word1和word2的距离并打印出来
    print(model.wv.distance('word1', 'word2'))
    
    # 计算word1和word2的余弦相似度并打印出来
    print(model.wv.similarity('word1', 'word2'))
