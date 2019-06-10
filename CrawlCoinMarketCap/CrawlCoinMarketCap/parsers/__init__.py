# -*- coding: utf-8 -*-
# !/usr/bin/python
import re
import time

import scrapy
import numpy as np
import pandas as pd


# from ..utils.write_excel import ExcelRW


class Parsers:
    # data = []
    # excel_path = './files/coins.xlsx'
    file_path = './files/'
    # 防止excel打开乱码
    encode = 'utf-8-sig'

    # excel = ExcelRW()

    # 解析解析虚拟货币币种列表
    def currencies_parser(self, response, allow_domain):
        # self.data = []
        print('=========⭐⭐⭐ Parsing list ⭐⭐⭐===========')

        items = response.css('.js-all-crypto-table tbody')
        # for item in items:
        href = items.css('.name a').xpath('./@href').getall()
        name = items.css('.name a::text').getall()
        symbol = items.css('.symb::text').getall()
        price = items.css('.price a::text').getall()
        market_value = items.css('.js-market-cap').xpath('./@data-value').getall()
        volume = items.css('.js-24h-volume').xpath('./@data-value').getall()
        transaction_share = items.css('.js-total-vol::text').getall()
        up_and_downs_24h = items.css('.js-currency-change-24h::text').getall()
        up_and_downs_7d = items.css('.js-currency-change-7d::text').getall()

        # self.data.append(
        #     [full_href, name, symbol, price, market_value, volume, transaction_share, up_and_downs_24h,
        #      up_and_downs_7d])

        # 保存为excel
        # header_data = [['名称', 20], ['符号', 20], ['价格', 10], ['市值', 40], ['成交量（24小时）', 10],
        #                ['交易份额', 10], ['涨跌（24小时）', 10], ['涨跌（7日）', 10]]
        # self.excel.save_excel('all list', self.data, self.excel_path, header_data)

        # 保存为csv文件
        data = {'链接': np.array(href), '名称': np.array(name), '符号': np.array(symbol), '价格': np.array(price),
                '市值': np.array(market_value), '成交量（24小时）': np.array(volume),
                '交易份额': np.array(transaction_share), '涨跌（24小时）': np.array(up_and_downs_24h),
                '涨跌（7日）': np.array(up_and_downs_7d)}
        data_frame = pd.DataFrame.from_dict(data, orient='index')

        # 用apply进行向量批量操作会比for循环快好几倍
        data_frame.iloc[:1] = data_frame.iloc[:1].apply(lambda uri: allow_domain + uri)
        data_frame = data_frame.transpose()
        data_frame.to_csv(self.file_path + 'coins_list.csv', encoding=self.encode, index=False, sep=',')

        return data_frame.iloc[:, 0].values

    # 解析单个币的历史数据请求，获取关键的curr_id和smlID用于获取单个币的所有历史数据
    def details_parser(self, response):
        whole_name = response.request.url.split('/')[-2]
        full_name = response.css('#fullColumn .float_lang_base_1::text').get()
        script_data = response.css('.fullHeaderTwoColumnPage--content > script:nth-of-type(2)').get()
        match_obj = re.findall(r'\d+', script_data)

        url = 'https://cn.investing.com/instruments/HistoricalDataAjax'
        headers = {
            'X-Requested-With': 'XMLHttpRequest',  # 异步请求的资源必须要加
        }
        form_data = {
            'curr_id': match_obj[0],
            'smlID': match_obj[1],
            'header': 'null',
            'st_date': '2010/05/08',
            'end_date': time.strftime("%Y/%m/%d", time.localtime()),
            'interval_sec': 'Daily',
            'sort_col': 'date',
            'sort_ord': 'DESC',
            'action': 'historical_data',
        }

        yield scrapy.FormRequest(url=url, formdata=form_data, callback=self.historical_parser,
                                 headers=headers, meta={'full_name': whole_name + ',' + full_name})

    # 解析单个币的所有历史数据
    def historical_parser(self, response):
        # self.data = []
        full_name = response.meta.get('full_name')
        results = response.css('#results_box tbody')
        # for item in results[:-1]:
        date = results.css('td:nth-child(1)::text').getall()
        close = results.css('td:nth-child(2)::text').getall()
        open = results.css('td:nth-child(3)::text').getall()
        high = results.css('td:nth-child(4)::text').getall()
        low = results.css('td:nth-child(5)::text').getall()
        volume = results.css('td:nth-child(6)::text').getall()
        up_down_percent = results.css('td:nth-child(7)::text').getall()

        # self.data.append([date, close, open, high, low, volume, up_down_percent])
        # header_data_coin = [['日期', 20], ['开盘', 20], ['收盘', 20],
        #                     ['高', 20], ['低', 20], ['交易量', 20], ['涨跌幅', 20]]
        # self.excel.save_excel(full_name, self.data, self.excel_path, self.header_data_coin)

        # 保存为csv文件
        data = {'日期': date, '开盘': open, '收盘': close, '高': high,
                '低': low, '交易量': volume, '涨跌幅': up_down_percent}
        data_frame = pd.DataFrame.from_dict(data, orient='index')
        data_frame = data_frame.transpose()
        data_frame.to_csv(self.file_path + full_name + '.csv', encoding=self.encode, index=False, sep=',')
