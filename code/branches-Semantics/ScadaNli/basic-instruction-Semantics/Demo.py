import sys
import os
from MainWindow import Ui_Dialog
from PyQt5.QtWidgets import QApplication, QDialog, QMenu
from PyQt5 import QtCore
from Record import *
from SpeechRecognition import *
from main import GetAttribution
from maps import *
import gensim



'''
Main window with 3 modules:
    1. Record Module ✔
    2. Dialog Module ✔
    3. Web Browser Module ✔ 
'''

'''
adding funtions:
    1. if some item in listWidget was double-clicked, 
        turn to the page corresponding to the item.
        
    2. clearing the temp audio file.
'''

text = ""
cache_path = "./cache/"
ip = "http://192.168.11.151/Scada/View.aspx?viewID="
'''
url form:
    http://192.168.11.151/Scada/View.aspx?viewID=2
    http://192.168.11.151/Scada/plugins/Chart/Chart.aspx?cnlNum=1&viewID=3&year=2020&month=2&day=14
    1. Views:
        "http://" + ip + "/Scada/View.aspx?viewID=" + view_id
        
    2. Tables:
        "http://" + ip + "/Scada/plugins/Chart/Chart.aspx?cnlNum=" + channel_id + "&year=%d&month=%d&day=%d" % y, m, d
    
'''

class MainWindow(QDialog, Ui_Dialog):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.RecBtn.pressed.connect(self.onPressed)
        self.RecBtn.released.connect(self.onReleased)
        self.listWidget.addItem("测试用Item1")
        self.listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.listWidget.customContextMenuRequested[QtCore.QPoint].connect(self.right_menu)
        self.t = GetAttribution()

    '''
    按住按钮开始录音
    '''
    def onPressed(self):
        self.rec = Recorder()
        self.begin = time.time()
        print("Start recording")
        self.rec.start()

    '''
    释放按钮停止录音
    '''
    def onReleased(self):
        # 停止录音
        global tag, text
        print("Stop recording")
        self.rec.stop()
        self.fina = time.time()
        t = self.fina - self.begin
        # 录音时长
        print("The recording time is %d sec" % t)

        # 保存.wav文件
        self.rec.save(cache_path+"tmp.wav")

        # 语音识别
        self.text = SpeechRecog(cache_path+"tmp.wav")

        # 显示语音识别结果到ListWeight上
        self.listWidget.addItem(self.text)

        # 自然语言处理
        self.od = self.t.NL2command(self.text)
        print(self.od)
        view_id = place_dict.get(self.od['location'])
        self.webEngineView_2.setUrl(QtCore.QUrl(ip + str(view_id)))


    '''
    listWeight中item的右键菜单
    '''
    def right_menu(self, pos):
        menu = QMenu(self.listWidget)
        opt = menu.addAction("删除")
        act = menu.exec_(self.listWidget.mapToGlobal(pos))
        current_item = self.listWidget.currentItem()
        if act == opt:
             self.listWidget.takeItem(self.listWidget.row(current_item))
        return



if __name__ == '__main__':
    # 初始化
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()

    # 退出
    sys.exit(app.exec_())
