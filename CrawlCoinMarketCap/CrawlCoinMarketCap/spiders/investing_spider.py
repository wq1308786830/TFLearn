# -*- coding: utf-8 -*-
# !/usr/bin/python
import scrapy

from ..parsers import Parsers


class InvestingSpider(scrapy.Spider):
    name = 'investing'

    allow_domains = ['https://cn.investing.com']

    url = 'https://cn.investing.com/crypto/currencies'

    parsers = Parsers()

    def parse(self, response):
        data = self.parsers.currencies_parser(response, self.allow_domains[0])

        for item in data:
            url = item[0] + '/historical-data' if item[0] else None
            if url:
                yield scrapy.Request(url=url, callback=self.parsers.details_parser)

    def start_requests(self):
        yield scrapy.Request(url=self.url, callback=self.parse)
