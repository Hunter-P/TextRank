# -*- encoding=utf-8 -*-
# author: hunter

import numpy as np
from collections import defaultdict
import thulac
import re
from sklearn.externals import joblib
from gensim.models import Word2Vec
import jieba.analyse
import jieba.posseg as pseg
import time
import jieba


class UndirectWeightedGraph:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, start, end, weight=1):
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self, iteration=10):
        """
        textrank算法的实现
        :param iteration: 迭代次数
        :return: dict type
        """
        print("begin to run rank func...")
        ws = defaultdict(float)
        outSum = defaultdict(float)  # 节点出度之和
        wsdef = 1.0 / (len(self.graph) or 1.0)  # 节点权值初始定义
        for n, edge in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((i[2] for i in edge), 0.0)
        sorted_keys = sorted(self.graph.keys())
        for i in range(iteration):  # 迭代
            # print("iteration %d..." % i)
            for n in sorted_keys:
                s = 0
                # 遍历节点的每条边
                for edge in self.graph[n]:
                    s += edge[2] / outSum[edge[1]] * ws[edge[1]]
                ws[n] = (1-self.d) + self.d*s  # 更新节点权值

        min_rank, max_rank = min(ws.values()), max(ws.values())
        # 归一化权值
        for n, w in ws.items():
            ws[n] = (w-min_rank/10) / (max_rank-min_rank/10)
        return ws


class TextRank:
    def __init__(self, data):
        """
        :param data: 输入的数据，字符串格式
        """
        self.data = data  # 字符串格式

    def extract_key_words(self, topK=20, window=4, iteration=200, allowPOS=('ns', 'n'), stopwords=True):
        """
        抽取关键词
        :param allowpos: 词性
        :param topK:   前K个关键词
        :param window: 窗口大小
        :param iteration: 迭代次数
        :param stopwords: 是否过滤停止词
        :return:
        """
        text = self.generate_word_list(allowPOS, stopwords)
        graph = UndirectWeightedGraph()
        # 定义共现词典
        cm = defaultdict(int)
        # 构建无向有权图
        for i in range(1, window):
            if i < len(text):
                text2 = text[i:]
                for w1, w2 in zip(text, text2):
                    cm[(w1, w2)] += 1
        for terms, w in cm.items():
            graph.add_edge(terms[0], terms[1], w)
        joblib.dump(graph, 'data/graph')
        ws = graph.rank(iteration)
        return sorted(ws.items(), key=lambda x: x[1], reverse=True)[:topK]

    def generate_word_list(self, allowPOS, stopwords):
        """
        对输入的数据进行处理，得到分词及过滤后的词列表
        :param allowPOS: 允许留下的词性
        :param stopwords: 是否过滤停用词
        :return:
        """
        s = time.time()
        # thu_tokenizer = thulac.thulac(filt=True, rm_space=True, seg_only=False)
        # text = thu_tokenizer.cut(self.data)
        text = [(w.word, w.flag) for w in pseg.cut(self.data)]  # 词性标注
        word_list = []
        if stopwords:
            stop_words = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
            stopwords_news = [line.strip() for line in open('stopwords_news.txt', encoding='UTF-8').readlines()]
            all_stopwords = set(stop_words + stopwords_news)
        # 词过滤
        if text:
            for t in text:
                if len(t[0]) < 2:
                    continue
                if len(t[0]) < 2 or t[1] not in allowPOS:
                    continue
                if stopwords:
                    # 停用词过滤
                    if t[0] in all_stopwords:
                        continue
                word_list.append(t[0])
        return word_list

    def extract_key_sentences(self, topK=3, window=3, ndim=20, allowPOS=('ns', 'ni', 'nl'), stopwords=True, iteration=300):
        """
        抽取关键句子
        :param topK: 前K句话
        :param window: 窗口大小
        :param ndim: 词向量维度
        :param allowPOS: 词性
        :param iteration: 迭代次数
        :param stopwords: 是否过滤停用词
        :return:
        """
        try:
            text = joblib.load("data/sentence_vectors")
        except FileNotFoundError:
            text = self.bulid_sentence_vec(ndim, allowPOS, stopwords)
        graph = UndirectWeightedGraph()
        # 构建无向有权图
        for i in range(1, window):
            if i < len(text):
                text2 = text[i:]
                for w1, w2 in zip(text, text2):
                    if not np.isnan(self.cos_sim(w1[1], w2[1])):
                        graph.add_edge(w1[0], w2[0], self.cos_sim(w1[1], w2[1]))
        ws = graph.rank(iteration)
        s = list(ws.keys())
        topK_sentences = sorted(ws.items(), key=lambda x: x[1], reverse=True)[:topK]
        s_w_index_list = [[i, s.index(i[0])] for i in topK_sentences]
        res = sorted(s_w_index_list, key=lambda x: x[1])
        return sorted(res, key=lambda x: x[1])

    def bulid_sentence_vec(self, ndim, allowPOS, stopwords):
        """
        构建句向量
        :param ndim: 词向量维度
        :param allowPOS: 词性
        :param stopwords: 是否过滤停用词
        :return:
        """
        print("bulid_sentence_vec")
        try:
            self.sentence_list = joblib.load("data/sentence_list")
            model = Word2Vec.load("model/w2v_model")
        except FileNotFoundError:
            self.sentence_list = self.generate_sentence_list(allowPOS, stopwords)
            model = self.bulid_w2c(ndim)
        sentence_vectors = [[sentence[0][0], self.sentence_vec(model, sentence[1], ndim)] for sentence in self.sentence_list]
        joblib.dump(sentence_vectors, "data/sentence_vectors")
        return sentence_vectors

    def sentence_vec(self, model, sentence, ndim):
        vec = np.zeros(ndim)
        count = 0
        for word in sentence:
            try:
                vec += model.wv[word]
                count += 1
            except KeyError as e:
                continue
        if count != 0:
            vec /= count
        return vec

    def bulid_w2c(self, ndim):
        """
        训练Wordvec模型
        :param ndim:  词向量维度
        :return:
        """
        print("train bulid_w2c...")
        data = [s[1] for s in self.sentence_list]
        model = Word2Vec(data, size=ndim, window=3, iter=10)
        model.save("model/w2v_model")
        return model

    def generate_sentence_list(self, allowPOS, stopwords):
        """
        对输入的数据进行处理，得到句子列表（包含原句和分词列表）
        :param stopwords: 是否过滤停用词
        :return:
        """
        sentence_list = [[i] for i in re.split(r"[.。?!！？]", self.data)]  # 分句
        # thu_tokenizer = thulac.thulac(rm_space=True)
        new_sentence_list = []
        if stopwords:
            stop_words = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
            try:
                stopwords_news = [line.strip() for line in open('stopwords_news.txt', encoding='UTF-8').readlines()]
                all_stopwords = stop_words + stopwords_news
            except:
                all_stopwords = stop_words
        else:
            all_stopwords = ''
        for s in sentence_list:
            # word_list = thu_tokenizer.cut(s[0])  # 分词
            word_list = [(w.word, w.flag) for w in pseg.cut(self.data)]  # 词性标注
            new_word_list = []
            # 过滤
            if word_list:
                for w in word_list:
                    if allowPOS and w[1] not in allowPOS:
                            continue
                    if stopwords and w[0] in all_stopwords:
                            continue
                    new_word_list.append(w[0])
            if new_word_list:
                new_sentence_list.append([s, new_word_list])
        return new_sentence_list

    @classmethod
    def cos_sim(cls, vec_a, vec_b):
        """
        计算两个向量的余弦相似度
        :param vec_a:
        :param vec_b:
        :return:
        """
        vector_a = np.mat(vec_a)
        vector_b = np.mat(vec_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        if cos == 'nan':
            print(cos)
        sim = 0.5 + 0.5 * cos
        return sim


# 测试， 对比jieba的 textrank, tf-idf算法
if __name__ == '__main__':
    content = '''今天小米双11可谓全家集体出动，除了手机，其他各条产品线也是一路狂奔，不断亮出耀眼数字刷屏。比如已经成为国内第一的小米电视，12个小时全渠道支付金额突破了10亿元，在天猫、京东、苏宁的销量、销售额全部都是第一，32寸、40寸、43寸、49寸、50寸、55寸、65寸七个单品额度也是销量第一。今天凌晨，小米电视更是只用9分02秒就入账1亿元，1小时58分到手5亿元。其他产品，截至中午12点50分，小米净水器全渠道支付金额破1亿元，创历史新高。截至16点20分，米家扫地机器人全渠道销售数量突破10万台，线下小米之家销量同比增长148倍，同时还是智能硬件单品单天最快破亿的。截至15点，天猫平台智能手环类目，小米手环3单品销量、销售额双第一。截止16点45分，天猫平台智能出行类目，九号平衡车单品销量、销售额双第一。手机方面，截至16点，小米MIX 3天猫、京东3000-4000元价位段销量第一，小米8天猫、京东、苏宁2000-3000元价位段销量第一，小米8青春版天猫、苏宁1000-2000元价位段销量第一。'''
    print(content)
    tr = TextRank(content)
    key_sentences = tr.extract_key_sentences(topK=4, window=3, ndim=10)
    print(key_sentences)
    key_words = jieba.analyse.textrank(content, topK=5, withWeight=True, allowPOS=('n', 'ni', 'nz'))
    print(key_words)
    key_words = jieba.analyse.extract_tags(content, topK=5, withWeight=True, allowPOS=('n', 'ni', 'nz'))   # tf-idf
    print(key_words)
    key_words = tr.extract_key_words(topK=5,  window=5, iteration=50, stopwords=True, allowPOS=('n', 'ni', 'nz'))
    print(key_words)

