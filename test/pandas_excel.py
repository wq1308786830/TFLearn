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
        locale.setlocale(locale.LC_NUMERIC, '')

        self.excel_path = file
        self.colordown = '#53c156'  # 跌颜色
        self.colorup = '#ff1717'  # 涨颜色
        self.data_mat = None

    # 读取数据并解析为可以直接绘图的数据
    def read_data(self):
        # 读取数据，sep=','表示每列以逗号间隔，thousands=','表示千分位的间隔符号位逗号
        data = pd.read_csv(self.excel_path, usecols=['日期', '开盘', '收盘', '高', '低', '交易量', '涨跌幅'], sep=',', thousands=',')
        data[data['交易量'] == 0] = np.nan
        data = data.dropna()
        data['日期'] = pd.to_datetime(data['日期'], format='%Y年%m月%d日')
        data['日期'] = data['日期'].apply(lambda x: date2num(x))
        data.sort_values(by='日期', ascending=True, inplace=True)
        data = data[['日期', '开盘', '收盘', '高', '低', '交易量', '涨跌幅']]
        self.data_mat = data.values

    def draw(self):
        self.read_data()
        fig, ax = plt.subplots(figsize=(1200 / 72, 480 / 72))
        fig.subplots_adjust(bottom=0.1)
        mpf.candlestick_ochl(ax, self.data_mat, colordown=self.colordown, colorup=self.colorup, width=0.3, alpha=1)
        ax.grid(True)
        ax.xaxis_date()
        plt.show()


tool = ShapeKLine('../files/EOS历史数据_历史行情,价格,走势图表_英为财情.csv')
tool.draw()
