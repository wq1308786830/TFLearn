# -*- coding: utf-8 -*-
# !/usr/bin/python
import re
import telnetlib
import time

import requests
import scrapy
import numpy as np
import pandas as pd


# from ..utils.write_excel import ExcelRW


class Parsers:
    file_path = './files/'
    # 防止excel打开乱码
    encode = 'utf-8-sig'

    HIGH_TYPE = '高匿'
    MAX_SECONDS = 1

    def proxy_parser(self, response):
        allow_domain = response.meta.get('allow_domain')
        proxies = response.css('#ip_list tr')
        print('开始过滤代理：')
        for proxy in proxies[1:]:
            ip = proxy.css('td:nth-child(2)::text').get().strip()
            port = proxy.css('td:nth-child(3)::text').get().strip()
            type_name = proxy.css('td:nth-child(5)::text').get().strip()
            speed = proxy.css('td:nth-child(7)>div.bar').attrib['title'].strip()
            speed_float = float(re.findall(r'\d+\.\d+', speed)[0])

            if type_name == self.HIGH_TYPE and speed_float <= self.MAX_SECONDS:
                proxy_https_url = 'https://' + ip + ':' + port
                try:
                    # telnet没有异常则是可用代理，最大不超过1.5秒
                    # telnetlib.Telnet(ip, port, timeout=1.5)
                    requests.get(allow_domain, proxies={'https': proxy_https_url}, timeout=1.5)
                    print('\n\n✨✨✨检测到可用代理✨✨✨', proxy_https_url)

                    return proxy_https_url
                except Exception as e:
                    print('不可用的代理地址：%s  原因：%s' % (proxy_https_url, e))

    # 解析解析虚拟货币币种列表
    def currencies_parser(self, response):
        print('=========⭐⭐⭐ Parsing list ⭐⭐⭐===========')

        allow_domain = response.meta.get('allow_domain')
        items = response.css('.js-all-crypto-table tbody')
        href = items.css('.name a').xpath('./@href').getall()
        name = items.css('.name a::text').getall()
        symbol = items.css('.symb::text').getall()
        price = items.css('.price a::text').getall()
        market_value = items.css('.js-market-cap').xpath('./@data-value').getall()
        volume = items.css('.js-24h-volume').xpath('./@data-value').getall()
        transaction_share = items.css('.js-total-vol::text').getall()
        up_and_downs_24h = items.css('.js-currency-change-24h::text').getall()
        up_and_downs_7d = items.css('.js-currency-change-7d::text').getall()

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

        coins_url = data_frame.iloc[:, 0].values
        for item in coins_url:
            url = item + '/historical-data' if type(item) == str else None
            if url:
                request = scrapy.Request(url=url, callback=self.details_parser)
                yield request

    # 解析单个币的历史数据请求，获取关键的curr_id和smlID用于获取单个币的所有历史数据
    def details_parser(self, response):
        whole_name = response.request.url.split('/')[-2]
        full_name = response.css('#fullColumn .float_lang_base_1::text').get().strip()
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

        request = scrapy.FormRequest(url=url, formdata=form_data, callback=self.historical_parser,
                                     headers=headers)
        request.meta['full_name'] = whole_name + ',' + full_name

        yield request

    # 解析单个币的所有历史数据
    def historical_parser(self, response):
        full_name = response.meta.get('full_name')
        results = response.css('#results_box tbody')
        date = results.css('td:nth-child(1)::text').getall()
        close = results.css('td:nth-child(2)::text').getall()
        open = results.css('td:nth-child(3)::text').getall()
        high = results.css('td:nth-child(4)::text').getall()
        low = results.css('td:nth-child(5)::text').getall()
        volume = results.css('td:nth-child(6)::text').getall()
        up_down_percent = results.css('td:nth-child(7)::text').getall()

        # 保存为csv文件
        data = {'日期': date, '开盘': open, '收盘': close, '高': high,
                '低': low, '交易量': volume, '涨跌幅': up_down_percent}
        data_frame = pd.DataFrame.from_dict(data, orient='index')
        data_frame = data_frame.transpose()
        data_frame.to_csv(self.file_path + full_name + '.csv', encoding=self.encode, index=False, sep=',')
