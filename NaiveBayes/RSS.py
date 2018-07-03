#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
import  feedparser
ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
print(ny['entries'])#'entries'是入口
