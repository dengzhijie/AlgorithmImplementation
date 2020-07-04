#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: test_numpy.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

import pandas as pd
import numpy as np
b=np.random.randn(1,5)
print(b.shape)
print(b.flatten())
print(b)
print(np.dot(b.T,b))