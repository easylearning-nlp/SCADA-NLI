import jieba
import sys
from gensim.models import Word2Vec




dict_path = './src/dict.txt'

def my_initialize():
    jieba.load_userdict(dict_path)



