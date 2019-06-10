# -*- coding: utf-8 -*-
# !/usr/bin/python
import re
import time

import scrapy

from ..utils.write_excel import ExcelRW


class Parsers:
    data = []
    excel_path = './files/coins.xlsx'
    excel = ExcelRW()
    header_data_coin = [['日期', 20], ['开盘', 20], ['收盘', 20],
                        ['高', 20], ['低', 20], ['交易量', 20], ['涨跌幅', 20]]

    # 解析解析虚拟货币币种列表
    def currencies_parser(self, response, allow_domain):
        self.data = []
        print('=========⭐⭐⭐ Parsing list ⭐⭐⭐===========')

        items = response.css('.js-all-crypto-table tbody tr')
        for item in items:
            href = item.css('.name a').xpath('./@href').get()
            full_href = (allow_domain + href) if href else ''
            name = item.css('.name a::text').get()
            symbol = item.css('.symb::text').get()
            price = item.css('.price a::text').get()
            market_value = item.css('.js-market-cap').xpath('./@data-value').get()
            volume = item.css('.js-24h-volume').xpath('./@data-value').get()
            transaction_share = item.css('.js-total-vol::text').get()
            up_and_downs_24h = item.css('.js-currency-change-24h::text').get()
            up_and_downs_7d = item.css('.js-currency-change-7d::text').get()

            self.data.append(
                [full_href, name, symbol, price, market_value, volume, transaction_share, up_and_downs_24h,
                 up_and_downs_7d])

        # 保存为excel
        header_data = [['名称', 20], ['符号', 20], ['价格', 10], ['市值', 40], ['成交量（24小时）', 10],
                       ['交易份额', 10], ['涨跌（24小时）', 10], ['涨跌（7日）', 10]]
        self.excel.save_excel('all list', self.data, self.excel_path, header_data)

        return self.data

    # 解析单个币的历史数据请求，获取关键的curr_id和smlID用于获取单个币的所有历史数据
    def details_parser(self, response):
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
                                 headers=headers, meta={'full_name': full_name})

    # 解析单个币的所有历史数据
    def historical_parser(self, response):
        self.data = []
        full_name = response.meta.get('full_name')
        results = response.css('#results_box tbody tr')
        for item in results[:-1]:
            date = item.css('td:nth-child(1)::text').get()
            close = item.css('td:nth-child(2)::text').get()
            open = item.css('td:nth-child(3)::text').get()
            high = item.css('td:nth-child(4)::text').get()
            low = item.css('td:nth-child(5)::text').get()
            volume = item.css('td:nth-child(6)::text').get()
            up_down_percent = item.css('td:nth-child(7)::text').get()

            self.data.append([date, close, open, high, low, volume, up_down_percent])

        self.excel.save_excel(full_name, self.data, self.excel_path, self.header_data_coin)
