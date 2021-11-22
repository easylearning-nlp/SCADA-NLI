# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(785, 599)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setHorizontalSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.webEngineView_2 = QtWebEngineWidgets.QWebEngineView(Dialog)
        self.webEngineView_2.setUrl(QtCore.QUrl("http://192.168.11.151/scada"))
        self.webEngineView_2.setObjectName("webEngineView_2")
        self.gridLayout.addWidget(self.webEngineView_2, 0, 1, 3, 1)
        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setObjectName("listWidget")
        self.gridLayout.addWidget(self.listWidget, 0, 0, 1, 1)
        self.RecBtn = QtWidgets.QPushButton(Dialog)
        self.RecBtn.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RecBtn.sizePolicy().hasHeightForWidth())
        self.RecBtn.setSizePolicy(sizePolicy)
        self.RecBtn.setMinimumSize(QtCore.QSize(23, 23))
        self.RecBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.RecBtn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.RecBtn.setObjectName("RecBtn")
        self.gridLayout.addWidget(self.RecBtn, 1, 0, 1, 1)
        self.gridLayout.setColumnStretch(0, 2)
        self.gridLayout.setColumnStretch(1, 5)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "SCADA-NLI"))
        self.RecBtn.setText(_translate("Dialog", "点击录音"))

from PyQt5 import QtWebEngineWidgets
