<<<<<<< .mine
请简要说明本文件夹的作用和组织方式
branches-Semantics：吴昊等开发语义依存分析下的自然语言接口

## 更新日志
### 20210520


---
##介绍
svn的各方法的说明、代码、数据和测试结果
本仓库代码以下几个部分，每部分按“软件结构”和“使用说明”介绍
1 . 自然语言指令分类代码
2 . 基于自然语言指令语义解析算法（TF-IDF+cos_sim）
3 . 复杂自然语言控制指令语义解析算法（基于语义依存句法分析的指令解析算法）
4 . 复杂自然语言查询指令语义解析算法（基于BERT的NL2SQL模型）
5. 对比实验算法
5.1 Template Matching算法
5.2 基于Seq2Seq理念的算法


### 环境配置
| 软件名称 | 版本 |
| PyCharm | community Edition 2019.3 |
| Python | 3.7 |
| pip | 版本 |
| ddparser | 0.1.1 |
| keras | 2.2.0 |
| keras-bert | 0.86.0 |
| Gensim | 3.8.1 |
| PyAudio | 0.2.11 |
| baidu-aip | 2.2.18.0 |
| Jieba | 0.39 |
| QtPy | 1.5.0 |
| baidu-aip | 2.2.18.0 |
| thrift | 0.13.0 |
| tensorflow | 2.1.0 |


#软件结构
### 自然语言指令分类
1、代码文件为‘KWECS.py’
使用关键词抽取与相似度最优化映射的办法

2、对比实验
关键词提取代码 'Keywordextraction.py'

余弦相似度代码 'cs.py'

对比实验结果
No.	Instruction Type	Cosine Similarity	Keyword Extraction	KWECS(ours)
1	Query	Basic	27/57	52/57	53/57
2		Complex	21/32	18/32	31/32
3	Control	Basic	21/34	23/34	32/34
4		Complex	58/77	69/77	77/77
Accuracy	63.5%(127/200)	81.0%(162/200)	96.5%(193/200)
MAX Time	0.85s	0.72s	0.74s


###基于自然语言指令语义解析算法（TF-IDF+cos_sim）

直接运行代码文件‘main.py’

采用论文提供的数据集进行相关测试，测试结果如下：

No.	Number	type	Accuracy	Precision	Recall	F-score	Max Time 
1	50	Query	93.47%	92.15%	94.25%	93.19%	0.82s
2	50	Control	94.15%	93.57%	92.14%	92.85%	0.74s
3	100	Query	90.14%	91.24%	91.71%	91.47%	0.67s
4	100	Control	91.25%	92.25%	92.14%	92.19%	0.89s
5	150	Query	88.91%	89.65%	88.54%	89.06%	0.87s
6	150	Control	88.15%	87.58%	89.91%	88.73%	0.73s
Avg	                                90.13%	90.37%	90.79%	90.58%	0.79S


###复杂自然语言控制指令语义解析算法（基于语义依存句法分析的指令解析算法）

使用百度开源框架DDparser为基础完成，在其依存分析结果上定义指令抽取规则

运行代码文件'NewParserControl.py'，算法包含提取算法和优化算法

使用数据集测试结果如下所示
No.	Number	Algorithms	              Accuracy	Precision	Recall	F-score	Max Time
1	50	Algorithm2	                87.24%	89.14%	87.64%	88.59%	1.421s
2		Algorithm2+Algorithm3	92.16% 	95.10%	93.27%	94.18%	1.468s
3	100	Algorithm2	                81.21%	80.09%	81.36%	80.66%	1.462s
4		Algorithm2+Algorithm3	90.10%	89.81%	90.65%	90.22 %	1.475s
5	150	Algorithm2	                77.90%	78.02%	79.37%	78.68%	1.468s
6		Algorithm2+Algorithm3	88.15%	87.58%	89.91%	88.73%	1.479s
Avg	                Algorithm2	                80.56%	80.56%	81.41%	81.49%	1.450s
	                Algorithm2+Algorithm3        89.47%	89.58%	90.72%	90.15%	1.474s


###复杂自然语言查询指令语义解析算法（基于BERT的NL2SQL模型）
运行代码文件'NL2SQL'
使用的是哈工大的BERT预训练模型
运行时间在1080ti上大约一天时间，baseline效果68.5%
实验效果如下所示：
NO.	Number	sel	agg	where	Max time
1	50	71.34%	74.47%	52.48%	1.78s
2	100	66.27%	68.69%	49.35%	1.89s
3	150	61.28%	62.28%	47.51%	1.83s
Avg	                64.62%	66.45%	48.95%	1.83s


###5.1 Template Matching算法
运行代码文件Template Matching.py

使用100条指令进行相关测试


###5.2 基于Seq2Seq理念的算法

使用的是Seq2Seq理念，编码器部分分别是RNN、LSTM、BiLSTM、BiLSTM-Attention

运行代码文件seq2seq.py

实验对比结果如下
No.&Number&	Accuracy&Precision&	Recall&	F-score&accuracy&MAX Time\\
\midrule   
1&	50&	96.27\%&	98.26\%&	97.37\%	&97.81\%&	96.00\%&2.07s\\
2&	100&	92.21\%&	94.62\%&	93.31\%&	93.96\%&	95.00\%&2.18s\\
3&	150&	89.21\%&	91.23\%&	90.39\%&	90.81\%&	94.67\%&3.29s\\
4&	200	&84.10\%&	85.23\%&	84.91\%&	85.07\%&	93.50\%&2.27s\\
\cline{2-8}
&\textbf{Avg}&\textbf{88.47\%}&\textbf{90.21\%}&\textbf{89.48\%}&\textbf{89.72\%}&\textbf{94.40\%}&\textbf{2.45s}\\

















































||||||| .r58
请简要说明本文件夹的作用和组织方式
branches-Semantics：吴昊等开发语义依存分析下的自然语言接口=======
branches-Semantics：吴昊等开发语义依存分析下的自然语言接口

## 更新日志
### 20210511
初建

---
## 介绍
svn的各方法的说明、代码、数据和测试结果
本仓库代码以下几个部分，每部分按“软件结构”和“使用说明”介绍。
1. DDParser解析自然语言接口算法
2. 各类seq2seq自然语言解析算法
3. NL2SQL数据库查询指令解析算法
4. Template Matching算法



### 环境配置
| 软件名称 | 版本 |
| - | - |
| PyCharm | community Edition 2019.3 |
| Jieba | 0.39 |
| QtPy | 1.5.0 |
| baidu-aip | 2.2.18.0 |


## 软件结构
### 1. DDParser解析自然语言接口算法
### 2. 各类seq2seq自然语言解析算法
### 3. NL2SQL数据库查询指令解析算法
### 4. Template Matching算法



## 使用说明
### 1. DDParser解析自然语言接口算法
### 2. 各类seq2seq自然语言解析算法
### 3. NL2SQL数据库查询指令解析算法
### 4. Template Matching算法

## 参与贡献
[吴昊]()

>>>>>>> .r60
