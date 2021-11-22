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
    text = ['打开', '关闭', '升高', '调低', '监测', '提高', '测量', '开启',
            '控制', '启动','预定','预约','放','关','抬','调到','调低','调高','关掉']
    t=wordextract()
    flag=0
    while True:
        sim=0
        instruction = input("user > ")
        starttime = time.time()
        list = t.Keyword_extrac(instruction)#没有使用关键词提取算法
        print(list)
        for w1 in text:
            for w2 in list:
                if w2 == w1:
                    flag=flag+1
                else:
                    flag=flag-1

        if flag >0:
            print("该指令是一条控制指令")
        else:
            print("该指令是一条查询指令")
        endtime = time.time()
        print(endtime-starttime)