import numpy as np
import pandas as pd

#1.增廣矩陣  [係數 | 常數]
A = np.array([
    [1.19, 2.11, -100, 1, 1.12],
    [14.2, -0.112, 12.2, -1, 3.44],
    [0, 100, -99.9, 1, 2.15],
    [15.3, 0.110, -13.1, -1, 4.16]
], dtype=float)

print("原始增廣矩陣:")
print(A)
print("\n")

n = A.shape[0]  # 方程數/未知數個數

# gaussian elimination
for k in range(n):
    # pivoting method
    maxindex = np.argmax(np.abs(A[k:n, k])) + k
       
    # 交換行（若需要）
    if maxindex != k:
        print(f"交換第{k+1}行和第{maxindex+1}行")
        A[[k, maxindex]] = A[[maxindex, k]]
        print(A)
        print("\n")
    
    # substract
    for row in range(k+1, n):
        multiplier = A[row, k] / A[k, k]
        A[row, k:] = A[row, k:] - multiplier * A[k, k:]
    
    print(f"第{k+1}輪:")
    print(A)
    print("\n")

# 回代求解
x = np.zeros(n)
for i in range(n-1, -1, -1):
    x[i] = (A[i, -1] - np.dot(A[i, i+1:n], x[i+1:n])) / A[i, i]

print("解向量 x:")
print(f"x1 = {x[0]}")
print(f"x2 = {x[1]}")
print(f"x3 = {x[2]}")
print(f"x4 = {x[3]}")
print("\n")


