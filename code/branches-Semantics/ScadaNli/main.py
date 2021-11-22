'''
主函数
@author：HaoyuPan
Date：2019年10月21日17:53:58
'''
import webbrowser
from jieba import load_userdict
from KeywordExtract import *
from init import my_initialize
from gensim.models import Word2Vec
from maps import *
import os


class GetAttribution():
    def __init__(self):
        load_userdict('./src/dict.txt')
        self.model = Word2Vec.load('./model/zhiwiki_news.word2vec')
        self.order_query = {
                    'time' : '',
                    'object' : '',
                    'location' : ''
        }
        self.order_control = {
            'time' : '',
            'object' : '',
            'location' : '',
            'action' : ''
        }


    def get_place(self, list):
        sim = 0
        word = ''
        for i in list:
            if i[0] in object_dict.keys():
                continue
            elif i[1] == 'ns':
                return i[0]
            else:
                for place in place_dict.keys():
                    t = self.model.wv.similarity(i[0], place)
                    if t > sim:
                        sim = t
                        word = place
        return word

    def get_object(self, list):
        sim = 0
        word = ''
        for i in list:
            if (i[1] == 'nz') or (i[1] == 'nc'):
                return i[0]
            else:
                for object in object_dict.keys():
                    if object not in self.model:
                        continue
                    t = self.model.wv.similarity(i[0], object)
                    if t > sim:
                        sim = t
                        word = object
        return word

    def classify(self, string):
        sim = 0
        word = ''
        for verb in control_dict.keys():
            t = self.model.wv.similarity(string, verb)
            if t > sim:
                sim = t
                word = verb
        return word

    def NL2command(self, string):
        l = []
        msg = string
        msg = del_stopwords(msg)
        msg = keyword_extract(msg, 'TF')
        attribution = word_tag(msg[1])
        print(attribution)
        # 通过动词判断是控制指令还是查询指令
        for i in attribution:
            # 构建动词列表
            if i[1] == 'v':
                l.append(i[0])

        # 若无动词，默认为查询指令
        if l == []:
            print('这是一条查询指令')
            self.order_query['time'] = msg[0]
            self.order_query['location'] = self.get_place(attribution)
            self.order_query['object'] = self.get_object(attribution)
            return self.order_query
        # 否则根据动词判断
        else:
            for i in l:
                t = self.classify(i)
            print(t)
            # 若字典中动词的映射是0，则为查询指令
            if control_dict.get(t) == 0:
                print('这是一条查询指令')
                self.order_query['time'] = msg[0]
                self.order_query['location'] = self.get_place(attribution)
                self.order_query['object'] = self.get_object(attribution)
                return self.order_query
            # 否则为控制指令
            else :
                print('这是一条控制指令')
                self.order_control['time'] = msg[0]
                self.order_control['location'] = self.get_place(attribution)
                self.order_control['object'] = self.get_object(attribution)
                self.order_control['action'] = t
                return self.order_control




if __name__ == '__main__':
    t = GetAttribution()
    while True:
        msg = input("me > ")
        if msg == 'quit':
            print("退出")
            continue
        else:
            print(t.NL2command(msg))
            # print(t.classify(msg))

