import StrThriftService1 #引入客户端类
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

try:
 #建立socket
 transport = TSocket.TSocket('192.168.43.244', 9090)
 #选择传输层，这块要和服务端的设置一致
 transport = TTransport.TBufferedTransport(transport)
 #选择传输协议，这个也要和服务端保持一致，否则无法通信
 protocol = TBinaryProtocol.TBinaryProtocol(transport)
 #创建客户端
 client = StrThriftService1.Client(protocol)
 transport.open()
#客户端send一段字符串给服务端
 print ("client - send：")
 send=input("请输入要发送的字符串：")
 msg = client.sendStr(send)+"12345"
 print ("测试：server收到的是否和发送的相同： " + msg)
 #客户端get服务端从键盘输入的字符串
 msg1=client.getStr("hello")+"111"
 print("client接收到server发送的字符串 :" +msg1 )
 #关闭传输
 transport.close()
#捕获异常
except Thrift.TException as ex:
 print ("%s" % (ex.message))