# TextRank
* 本代码实现了基于TextRank算法的热词提取和自动文摘功能。
* 具备停止词和词性过滤功能，可自定义新的停止词词库，命名为stopwords_news.txt，格式与stopwords.txt相同。
* 分词和词性标注基于jieba。
* 对于自动文摘，相邻句子相似度采用word2vec句向量的余弦距离，因此速度会比基于共现词频的方法要慢一些。

Any problem happened when using the codes, please contact me.

