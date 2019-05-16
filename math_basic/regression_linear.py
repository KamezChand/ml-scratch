# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt("click.csv",delimiter=",",skiprows=1)
# -

train_x = train[:,0]
train_y = train[:,1]

mu = train_x.mean()
sigma = train_x.std()

print(mu,sigma)

def standardize(x):
    return (x - mu) / sigma

train_x = standardize(train_x)

def to_matrix(x):
    return np.vstack([np.ones(x.size),x,x**2]).T

to_matrix(train_x)

theta = np.random.rand(3)

theta

def f(x):
    return np.dot(x,theta)

X = to_matrix(train_x)

f(X)

def E(x,y):
    return 0.5*np.sum((y - f(x))**2)

E(X,train_y)

f(X)

# +
# 学習率
ETA = 1e-3
# 誤差の差分
diff = 1
# 更新回数
count = 0
# 誤差の差分が0.01以下になるまでパラメータ更新を繰り返す
error = E(X, train_y)
while diff > 1e-2:
    # 更新結果を一時変数に保存
    theta = theta - ETA * np.dot(f(X) - train_y, X)

    # 前回の誤差との差分を計算
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

    # ログの出力
    count += 1
    log = '{}回目: theta = {}, 差分 = {:.4f}'
    print(log.format(count, theta, diff))

# プロットして確認
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()
# -

E_current,E_new


