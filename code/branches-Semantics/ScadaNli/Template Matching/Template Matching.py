'''
复杂操控自然语言指令主函数
@author：wuhao
Date：2020年11月27日13:53:48
'''
from ddparser import  DDParser
from maps import *
import jieba
import jieba.posseg as pseg
import StrThriftService1 #引入客户端类
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
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
        ParserDict = sentence[0]
        relationlist = ParserDict['deprel']
        postaglist = ParserDict['postag']
        wordlist = ParserDict['word']
        RESULT = []
        for relation in relationlist:
            number = number+1
            if relation == 'ATT':
                if postaglist[number-1] == 'v':
                    Complex_control['action'] = wordlist[number-1]
                else:
                    Complex_control['action'] = wordlist[number-2]
                Complex_control['location'] = wordlist[number]
            elif (relation == 'VOB') or (relation =='COO'):
                Complex_control['object'] = wordlist[number]
                RESULT.append(Complex_control.copy())
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
        else:
            od = w.ControlParser(NlControl)
            print(od)
            Parameter = []
            dictControl = []
            for nl in od:
                actionpParameters = control_dict.get(nl['action'])
                view_id = place_dict.get(nl['location'])
                cnl_num = object_dict.get(nl['object'])
                #Parameter=[actionpParameters,view_id,cnl_num]
                Parameter = [actionpParameters, view_id]
                dictControl.append(Parameter.copy())
            print(dictControl)
            for i in dictControl:
                try:
                    # 建立socket
                    transport = TSocket.TSocket('192.168.43.244', 9090)
                    # 选择传输层，这块要和服务端的设置一致
                    transport = TTransport.TBufferedTransport(transport)
                    # 选择传输协议，这个也要和服务端保持一致，否则无法通信
                    protocol = TBinaryProtocol.TBinaryProtocol(transport)
                    # 创建客户端
                    client = StrThriftService1.Client(protocol)
                    transport.open()
                    # 客户端send一段字符串给服务端
                    # print("client - send：")
                    send =str(i[1])+" "+str(i[0])
                    msg = client.sendStr(send)

                    # 关闭传输
                    transport.close()
                    time.sleep(3)

                # 捕获异常
                except Thrift.TException as ex:
                    print("%s" % (ex.message))