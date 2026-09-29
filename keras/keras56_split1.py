# 시계열 데이터를 지정한 timestep으로 자르는 코드

import numpy as np

a = np.array(range(1, 11))
print(a) # [ 1  2  3  4  5  6  7  8  9 10]
print(a.shape) # (10,)

size = 5 # timestep 사이즈

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): # 10-3+1 = range(0, 8)
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)


print(split_x(a, 3))
# [[ 1  2  3]
#  [ 2  3  4]
#  [ 3  4  5]
#  [ 4  5  6]
#  [ 5  6  7]
#  [ 6  7  8]
#  [ 7  8  9]
#  [ 8  9 10]]
print(split_x(a, 3).shape) # (8, 3)


bbb = split_x(a, 5)
print(bbb)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
print(bbb.shape) # (6, 5)


# x와 y 분리하기. 방법1
x = bbb[:, :-1]
y = bbb[:, -1]
print(x)
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]
print(x.shape) # (6, 4)
print(y) # [ 5  6  7  8  9 10]
print(y.shape) # (6,)


# x와 y 분리하기. 방법2
x = np.array([i[:-1] for i in bbb])
y = np.array([i[-1] for i in bbb])
print(x)
print(x.shape) # (6, 4)
print(y)
print(y.shape) # (6,)


# # x와 y 분리하기. 방법3
x = []
for i in bbb:
    x.append(i[:-1])
x = np.array(x)

y = []
for i in bbb:
    y.append(i[-1])
y = np.array(y)

print(x)
print(x.shape) # (6, 4)
print(y)
print(y.shape) # (6,)