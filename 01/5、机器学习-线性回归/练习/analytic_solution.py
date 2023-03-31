#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/31 11:46
# @Author  : 蓝桉
# @File    : analytic_solution.py
# @Software: PyCharm
# pip install -i https://pypi.doubanio.com/simple/ 包名
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(40)
X_1 = np.random.randn(100,1)
X_2 = np.random.randn(100,1)
X_b = np.c_[np.ones((100,1)),X_1,X_2]
y = 4 + 3*X_1 + 5*X_2 + np.random.randn(100,1)
print(X_b)
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta)

X_new = np.array([[0,0],
                  [2,3]])
X_new_b = np.c_[np.ones((2,1)),X_new]
print(X_new_b)
y_predict = X_new_b.dot(theta)
print(y_predict)

plt.plot(X_new[:,0],y_predict,'r-')
plt.plot(X_1,y,'b.')
plt.axis([0,2,0,25])
plt.show()