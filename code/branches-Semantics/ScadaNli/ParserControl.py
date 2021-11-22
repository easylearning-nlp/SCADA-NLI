'''
复杂操控自然语言指令主函数
@author：wuhao
DateStart：2020年12月15日13:53:48
'''
from ddparser import  DDParser
import jieba
import jieba.posseg as pseg
import time

class ControlTree():
    def DataProcess(self,text):
        words = pseg.cut("百度一个家科技公司")
        jieba.enable_paddle()#启动paddle模式
        words = pseg.cut(text,use_paddle=True)#paddle模式
        for word , flag in words:
            print("%s %s"%(word,flag))

    def ControlParser(self,sentence):
        Complex_control = {
            'action': '',
            'location': '',
            'object': ''
        }

        ddp = DDParser(use_pos=True)
        sentence = ddp.parse(sentence)
        number = -1
        flag = 0
        ParserDict = sentence[0]
        relationlist = ParserDict['deprel']
        postaglist = ParserDict['postag']
        wordlist = ParserDict['word']
        headlist = ParserDict['head']
        RESULT = []
        object_value = []  # 将object对应的关键信息做一个筛选，n和nz以外的删除
        for relation in relationlist:
            number = number+1
            if relation == 'ATT':
                '''
                利用ATT关系遍历前方动词
                if postaglist[number-1] == 'v':
                    Complex_control['action'] = wordlist[number-1]
                elif postaglist[number-2] == 'v' :
                    Complex_control['action'] = wordlist[number-2]
                else:
                    flag
                '''
                Complex_control['location'] = wordlist[number]
                Complex_control['object'] = wordlist[headlist[number]-1]
            if (relation == 'VOB')or(relation=='COO'):
                if relation =='VOB':
                    Complex_control['action'] = wordlist[headlist[number] - 1]
                else:
                    Complex_control['object'] = wordlist[number]
                #结果优化算法
                if (Complex_control['action'] != '') and (Complex_control['location'] != '') and (Complex_control['object'] != ''):
                    RESULT.append(Complex_control.copy())

        for item in RESULT:
            object_value.append(item['object'])
        i = 0
        for string1 in wordlist:
            i = i+1
            for string2 in object_value:
                if string2 == string1:
                    if (postaglist[i - 1] != 'nz') and (postaglist[i - 1] != 'n'):
                        RESULT = list(filter(lambda x: x['object'] != str(string2), RESULT))
        return RESULT
'''
        parameters=[]
        for nl in new:
            actionpParameters = control_dict.get(nl['action'])
            view_id = place_dict.get(nl['location'])
            cnl_num = object_dict.get(nl['object'])
            parameters=[actionpParameters,view_id,cnl_num]
        return  parameters
'''
if __name__=='__main__':
    w=ControlTree()
    while True:
        NlControl = input("user > ")
        if NlControl == 'quit':
            print("退出")
            continue
        elif w.ControlParser(NlControl) == []:
            print("不好意思，请您输入主动句式")
        else:
            starttime = time.time()
            od = w.ControlParser(NlControl)
            endtime = time.time()
            alltime = endtime - starttime
            print(alltime)
            print(od)