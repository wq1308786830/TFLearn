# -*- coding: utf-8 -*-
# !/usr/bin/python
import scrapy

from ..parsers import Parsers


class InvestingSpider(scrapy.Spider):
    name = 'investing'

    proxy_https_url = None
    meta = {'proxy': proxy_https_url}  # 设置ip代理

    allow_domains = ['https://cn.investing.com']

    start_urls = ['https://www.xicidaili.com/wn/', 'https://cn.investing.com/crypto/currencies']

    parsers = Parsers()

    def parse(self, response):
        self.proxy_https_url = self.parsers.proxy_parser(response)
        if self.proxy_https_url is None:
            return
        else:
            self.meta['proxy'] = self.proxy_https_url

        yield scrapy.Request(url=self.start_urls[1], callback=self.parsers.currencies_parser, meta=self.meta)

    def start_requests(self):
        yield scrapy.Request(url=self.start_urls[0], callback=self.parse)
