#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
from numpy import *
from matplotlib import pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf8')
x = arange(-10,10,0.01)

y = 1/(1+exp(-x))
plt.title('Sigmod Function')
plt.plot(x,y,'b',linewidth = '5')
plt.show()

