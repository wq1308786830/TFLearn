# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class CrawlcoinmarketcapItem(scrapy.Item):
    # define the fields for your item here like:
    name = scrapy.Field()
    symbol = scrapy.Field()
    marketValue = scrapy.Field()
    volume = scrapy.Field()
    transactionShare = scrapy.Field()
    upAndDowns24H = scrapy.Field()
    upAndDowns7D = scrapy.Field()
