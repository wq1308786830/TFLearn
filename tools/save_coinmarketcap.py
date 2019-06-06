#!/usr/bin/env python
# -*- coding: utf-8 -*-

from coinmarketcap import Market

coins = Market()
coin_list = []
data = coins.ticker()
for coin in data['data']:
    print(coin)
    coin_list.append(coin['symbol'])
