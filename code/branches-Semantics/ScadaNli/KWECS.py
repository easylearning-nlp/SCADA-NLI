'''
主函数
分类算法
@author：wuhao
Date：2021年1月24日12:08:58
'''
import gensim
from gensim.models import KeyedVectors
import jieba.analyse
import jieba
import time

class wordextract():
    def Keyword_extrac(self,text):
        res = []
        keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=True, allowPOS=('v','vn'))
        for item in keywords:
            res.append(item[:][0])
        return res

if __name__ == '__main__':
    model = gensim.models.Word2Vec.load('C:/Users/Mr、key/Desktop/SCADANLI2020/CodeSCADANLI/keyvaluemaps/model/zhiwiki_news.word2vec')
    text = ['打开', '关闭', '升高', '调低', '监测', '提高', '测量', '开启',
            '控制', '启动','预定','预约','放','关','抬','调到','调低','调高','关掉']
    t=wordextract()
    while True:
        sim=0
        instruction = input("user > ")
        starttime = time.time()
        list = t.Keyword_extrac(instruction)
        print(list)
        for w2 in list:
            if w2 not in model:
                print("该指令是一条控制指令")
                continue
        for w1 in text:
            for w2 in list:
                cossim=model.wv.similarity(w1, w2)
                if cossim>sim:
                    sim=cossim
        print(sim)

        if sim >= 0.65:
            print("该指令是一条控制指令")
        else:
            print("该指令是一条查询指令")
        endtime = time.time()
        print(endtime-starttime)

