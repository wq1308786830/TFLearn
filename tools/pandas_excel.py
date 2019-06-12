#!/usr/bin/env python
# -*- coding: utf-8 -*-
import locale

import matplotlib.pyplot as plt
import mpl_finance as mpf
import numpy as np
import pandas as pd
from matplotlib.pylab import date2num
from pandas.plotting import register_matplotlib_converters

'''
绘制k线图
'''


class ShapeKLine:

    # @file文件路径
    def __init__(self, file):
        # 注册带千分位逗号间隔的数字字符串转型float，方便处理数字（不使用register_matplotlib_converters会报警告）
        register_matplotlib_converters()
        # 千分位逗号间隔的数字字符串转型float
        locale.setlocale(locale.LC_NUMERIC, '')

        self.excel_path = file
        self.colordown = '#53c156'  # 跌颜色
        self.colorup = '#ff1717'  # 涨颜色
        self.data = None
        self.data_mat = None

    # 读取数据并解析为可以直接绘图的数据
    def read_data(self):
        # 读取数据，sep=','表示每列以逗号间隔，thousands=','表示千分位的间隔符号位逗号转为numeric数据
        self.data = pd.read_csv(self.excel_path, usecols=['日期', '开盘', '收盘', '高', '低', '交易量', '涨跌幅'], sep=',',
                                thousands=',')
        self.data[self.data['交易量'] == 0] = np.nan
        self.data = self.data.dropna()
        self.data['日期'] = pd.to_datetime(self.data['日期'], format='%Y年%m月%d日')
        self.data['日期'] = self.data['日期'].apply(lambda x: date2num(x))
        # 没有千分位的不会自动转化为numeric类型，需要手动采用向量操作提速
        self.data[['开盘', '收盘', '高', '低']] = self.data[['开盘', '收盘', '高', '低']].apply(pd.to_numeric)
        self.data.sort_values(by='日期', ascending=True, inplace=True)
        self.data = self.data[['日期', '开盘', '收盘', '高', '低', '交易量', '涨跌幅']]
        self.data_mat = self.data.to_numpy()

    def draw(self):
        self.read_data()
        fig, ax = plt.subplots(figsize=(1200 / 72, 480 / 72))
        fig.subplots_adjust(bottom=0.1)
        mpf.candlestick_ochl(ax, self.data_mat, colordown=self.colordown, colorup=self.colorup, width=0.3, alpha=1)
        ax.grid(True)
        ax.xaxis_date()
        plt.show()

# 打开以下注释运行
# tool = ShapeKLine('../CrawlCoinMarketCap/files/eos,EOS.csv')
# tool.draw()
