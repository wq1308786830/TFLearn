#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import mpl_finance as mpf
import numpy as np
import pandas as pd
from matplotlib.pylab import date2num

excel_path = '../files/Bitcoin - 比特币历史数据_历史行情,价格,走势图表_英为财情.csv'

data = pd.read_csv(excel_path, usecols=['日期', '开盘', '收盘', '高', '低', '交易量'])
data[data['交易量'] == 0] = np.nan
data = data.dropna()
data.sort_values(by='日期', ascending=True, inplace=True)
data = data[['日期', '开盘', '收盘', '高', '低', '交易量']]
# data=data.head(60)

data.date = pd.to_datetime(data['日期'])
data.date = data.date.apply(lambda x: date2num(x))
data_mat = data.as_matrix()

fig, ax = plt.subplots(figsize=(1200 / 72, 480 / 72))
fig.subplots_adjust(bottom=0.1)
mpf.candlestick_ochl(ax, data_mat, colordown='#53c156', colorup='#ff1717', width=0.3, alpha=1)
ax.grid(True)
ax.xaxis_date()
plt.show()
