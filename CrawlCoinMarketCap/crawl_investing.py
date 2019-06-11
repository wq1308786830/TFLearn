# -*- coding: utf-8 -*-
# !/usr/bin/python

from scrapy import cmdline


name = 'investing'
# 查看log级别
# https://docs.scrapy.org/en/latest/topics/logging.html#topics-logging-levels
cmd = 'scrapy crawl {0} --loglevel DEBUG'.format(name)
cmdline.execute(cmd.split())
