# Scada-NLI

## 更新日志
### 20201224
增加了依存指令解析代码
运行ParserControl.py

### 20200429
上传100条测试用例和测试结果表格`04.系统测试与验收\测试用例.xlsx`

### 20200425
增加了对指令分类的算法。

### 20200420
1. 修改`ScadaNli/Demo.py`中代码，将其中的函数全部封装到一个`GetAttribution`类中，解决了在Demo中无法读取model对象的问题。
2. 修改时间提取格式为 *时:分:秒*


### 20200412
1. 简化了`ScadaNli/Demo.py`中关键词匹配的代码，把相关的代码转移到了`ScadaNli/main.py`中；
2. 优化了相似度计算算法，与maps字典内的词语计算相似度；
3. 对自然语言进行词频排序后，不再返回一个字符串，而是`[[词语, 词性], [词语, 词性], ...]`的列表。


### 20200408
1. 在README中添加了关于PyQt中Designer工具的相关说明和对Gensim中Word2vec训练模型文件的几点说明；
2. 在`word2vec/test.py`中添加了计算词语欧式距离的代码；
3. 修复了`ScadaNli/Demo.py`中词义相似度不匹配的BUG。


---

## 介绍
本仓库代码共分为两个部分：
1. 词嵌入算法word2vec训练代码
2. 自然语言接口设计代码



### 环境配置
| 软件名称 | 版本 |
| - | - |
| PyCharm | community Edition 2019.3 |
| Python | 3.7 |

| pip | 版本 |
| - | - |
| Gensim | 3.8.1 |
| PyAudio | 0.2.11 |
| Jieba | 0.39 |
| QtPy | 1.5.0 |
| baidu-aip | 2.2.18.0 |
| ddparser | 0.1.1 |


## 软件结构
### word2vec训练
因此本节词向量的训练分为以下两个部分来进行：
+ 对中文语料进行预处理。
+ 利用gensim库进行词向量的训练。

1. 语料库预处理
相关代码文件为`data_pre_process.py`和`langconv.py`
采用的是维基百科里面的中文网页作为训练的语料库。将下载好的数据放置在`word2vec/data`文件夹里面。对中文维基百科语料库的具体操作分为以下几步：

+ 将维基百科语料库xml格式转化为txt格式。
+ 将繁体字转化为简体字。
+ 利用jieba分词进行中文分词。

2. 词向量训练
代码文件为`training.py`
采用的是gensim库，使用Google体出的[word2vec算法](https://arxiv.org/pdf/1301.3781.pdf)进行词向量的训练，参数说明：

+ sg：sg可赋值为0或1。其中，其中sg=1表示[Skip-gram模型](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)，sg=0表示[CBOW模型](https://en.wikipedia.org/wiki/Bag-of-words_model)。本实验经过对比，发现Skip-gram模型的效果（即sg=1）的效果要优于CBOW。
+ size：表示词向量的维度。本实验设置size=150。
+ window:表示当前词与预测词之间的可能的最大距离。本实验设置为5。
+ min_count：表示过滤出现的次数过少词的阈值，一个词语出现次数小于该值，直接忽略。
+ workers：表示训练词向量时使用的线程数，本文设置为9。

训练好的模型一般包含三个文件。由于训练数据的大小不同，生成的模型文件个数、大小也对应不同。

关于生成文件的说明，可以参考谷歌论坛上的[这篇帖子的回答](https://groups.google.com/forum/#!topic/gensim/bpOSS-PDl_4)。


3. 模型测试

代码文件为`test.py`
此代码用于测试模型效果，提供如下几种方法：
+ ```model = gensim.models.Word2Vec.load('model_file')``` 读取模型
+ ```model.wv.similarity('word1', 'word2')``` 计算word1和word2的余弦相似度
+ ```model.wv.distance('word1', 'word2')``` 计算word1和word2的欧式距离
+ ```model.wv.most_similar(positive='word', topn=n)``` 查询与word相似度最高的n个词，返回一个列表

更多使用方法请参考[gensim官方文档](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec)


### 自然语言接口


1. 录音

实现该功能的代码文件`SCADA_NLI/Record.py`

创建Recorder类，成员方法包括start()、__recording()、stop()、save()。
分别用于开始录音、创建录音线程、停止录音和保存录音文件。音频文件保存在`SCADA_NLI/cache`中。

2. 百度语音识别接口调用

实现该功能的代码文件`SCADA_NLI/SpeechRecognition.py`

通过将音频文件.wav上传百度AI平台的语音识别接口，返回JSON格式识别结果。读取JSON的"result"字段获取文本。

3. 自然语处理模块

    **3.1 初始化**

    初始化代码文件`SCADA_NLI/init.py`

    用于读取用户自定义字典

    **3.2  关键词提取**

    关键词提取代码文件`SCADA_NLI/KeywordExtract.py`

    关键词提取实现了对文本的分词、停止词过滤、根据词性提取关键词和将关键词与中间语言的对应属性进行匹配。

    **3.3 映射字典**

    代码文件`SCADA_NLI/maps.py`

    该部分代码用于存储关键词-SCADA界面的http地址映射关系。其中http地址需要根据本机SCADA的服务器地址修改。

4. GUI

GUI用PyQT包实现，实现该功能的代码文件 `SCADA_NLI/MainWindow.ui`和`SCADA_NLI/MainWindow.py`。
其中，`MainWindow.ui`为Qt窗体样式文件；`MainWindow.py`为Python实现代码。文件`SCADA_NLI/MainWindow.ui`需要通过PyQt库中的Designer应用程序打开，并且通过Pyuic转化成Python脚本。

关于这两个过程在Pycharm中的设置，请参考[这篇博客](https://blog.csdn.net/px41834/article/details/79383985)

5. 演示

实现该功能的代码文件是`SCADA_NLI/Demo.py`

Demo给GUI添加了事件响应方法，实现了录音、语音识别、自然语言处理和界面展示的功能。

## 使用说明

1. 需要演示图形界面，直接运行`SCADA-NLI/Demo.py`。

    其中，word2vec训练模型文件需要放置`SCADA-NLI/model`文件夹中。  

2. word2vec训练：

    第一步运行`data_pre_process.py`，对原始的中文维基百科语料进行预处理，该代码运行后会在`word2vec/data`文件夹下载维基语料库，保存为`reduce_zhiwiki.txt`文件（大约为1.2G）。
    
    如果需要修改语料，可以直接修改文件`reduce_zhiwiki.txt`。如需要快速检验训练模型的效果，可以删除`reduce_zhiwiki.txt`文件内容，减少大小，以便快速训练。

    如果需要自建语料库，需要将收集的文本保存为文本文件`your_corpus.txt`。通过分词代码分词后，还需要修改`training.py`和`test.py`中对应的文件名。

    第二步运行`training.py`，开始训练。训练时间较长，取决于语料库的大小。

    最后，可以通过`test.py`检验模型的效果。

3. 自建语料库步骤

    如果需要自建语料库，我们需要将特定的语料库保存到txt文本中，然后运行`words_cut.py` 采用jieba分词并且加载停用词表（stopwords.txt）将需要的语料库变为和维基百科语料库一样的格式。并且将分词后的文本写入到`text_split_no.txt`中去。
    
    最后，接下来我需要通过运行`union.py`将`text_split_no.txt`和`reduce_zhiwiki.txt`合并到一起。**注意在python代码中修改相应的路径和文件名**。


## 参与贡献
[潘浩宇]()
[吴昊]()
