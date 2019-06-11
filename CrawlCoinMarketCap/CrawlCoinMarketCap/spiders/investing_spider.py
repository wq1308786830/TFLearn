# -*- coding: utf-8 -*-
# !/usr/bin/python
import scrapy

from ..parsers import Parsers


class InvestingSpider(scrapy.Spider):
    name = 'investing'

    allow_domains = ['https://cn.investing.com']

    start_urls = ['https://www.xicidaili.com/wn/', 'https://cn.investing.com/crypto/currencies']

    parsers = Parsers()

    def parse(self, response):
        proxy_https_url = self.parsers.proxy_parser(response)

        request = scrapy.Request(url=self.start_urls[1], callback=self.parsers.currencies_parser)
        request.meta['proxy'] = proxy_https_url
        request.meta['allow_domain'] = self.allow_domains[0]

        yield request

    def start_requests(self):
        yield scrapy.Request(url=self.start_urls[0], callback=self.parse)
