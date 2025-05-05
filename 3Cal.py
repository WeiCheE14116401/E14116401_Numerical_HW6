import numpy as np

def crout_tridiagonal(A, b):    #使用Crout解tri-diagonal system (Ax = b)

    n = len(b)
    # 初始化L和U矩陣
    L = np.zeros((n, n))
    U = np.eye(n)
    
    # 計算L和U的元素
    L[0, 0] = A[0, 0]
    U[0, 1] = A[0, 1] / L[0, 0] if n > 1 else 0
    
    for i in range(1, n-1):
        L[i, i-1] = A[i, i-1]  # m_i
        L[i, i] = A[i, i] - L[i, i-1] * U[i-1, i]  # l_i
        U[i, i+1] = A[i, i+1] / L[i, i]  # u_i

    if n > 1:
        L[n-1, n-2] = A[n-1, n-2]  # m_n
        L[n-1, n-1] = A[n-1, n-1] - L[n-1, n-2] * U[n-2, n-1]  # l_n
        
    print("\nL矩陣:")
    print(L)
    print("\nU矩陣:")
    print(U)
       
    # 求解Ly = b (前代法)
    y = np.zeros(n)
    y[0] = b[0] / L[0, 0]
    
    print("\n前代法求解Ly = b")
    print(f"y_1 = {y[0]}")
    
    for i in range(1, n):
        y[i] = (b[i] - L[i, i-1] * y[i-1]) / L[i, i]
        print(f"y_{i+1} = {y[i]}")
    
    # 求解Ux = y (回代法)
    x = np.zeros(n)
    x[n-1] = y[n-1]
    
    print("\n回代法求解Ux = y")
    print(f"x_{n} = {x[n-1]}")
    
    for i in range(n-2, -1, -1):
        x[i] = y[i] - U[i, i+1] * x[i+1]
        print(f"x_{i+1} = {x[i]}")
    return x

# 定義三對角矩陣A和右側向量b
A = np.array([
    [3, -1, 0, 0],
    [-1, 3, -1, 0],
    [0, -1, 3, -1],
    [0, 0, -1, 3]
], dtype=float)

b = np.array([2, 3, 4, 1], dtype=float)

print("原始三對角矩陣A:")
print(A)
print("\n右側向量b:")
print(b)

# 求解系統
x = crout_tridiagonal(A, b)

# 顯示最終解
print("\n== 最終解 ==")
for i, xi in enumerate(x):
    print(f"x_{i+1} = {xi}")

