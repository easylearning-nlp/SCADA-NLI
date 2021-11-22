# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
import codecs

#from matplotlib.pyplot import plt


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def my_function():
    wiki_news = codecs.open('./data/reduce_zhiwiki.txt','r', encoding='utf-8')
    model = Word2Vec(LineSentence(wiki_news), sg=1, size=150, window=5, min_count=5, workers=9, compute_loss=True)
    print(model.get_latest_training_loss)
    model.save('zhiwiki_news.word2vec')


if __name__ == '__main__':
    my_function()