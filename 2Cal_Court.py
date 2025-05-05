import numpy as np

# 定義矩陣A
A = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
], dtype=float)

print("原始矩陣A:")
print(A)
print("\n")

# 計算帶寬
n = A.shape[0]
lower_bw = max(i-j for i in range(n) for j in range(n) if A[i,j] != 0 and i>=j)
upper_bw = max(j-i for i in range(n) for j in range(n) if A[i,j] != 0 and j>=i)
bandwidth = max(lower_bw, upper_bw)

# Crout Decomposition
def crout_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.eye(n) 
    
    for j in range(n):
        # 計算L的第j列
        for i in range(j, n):
            sum_term = sum(L[i, k] * U[k, j] for k in range(j))
            L[i, j] = A[i, j] - sum_term
            print(f"L[{i+1},{j+1}] = {L[i,j]:.6f}")
        
        # 計算U的第j行
        for i in range(j+1, n):
            sum_term = sum(L[j, k] * U[k, i] for k in range(j))
            U[j, i] = (A[j, i] - sum_term) / L[j, j]
            print(f"U[{j+1},{i+1}] = {U[j,i]:.6f}")
    
    return L, U

# LU分解求逆矩陣
def inverse_with_crout(A):
    n = A.shape[0]
    L, U = crout_decomposition(A)
    A_inv = np.zeros((n, n))
    
    for j in range(n):
        print(f"\n求解逆矩陣第{j+1}列:")
        # 解Ly = e_j
        e = np.zeros(n)
        e[j] = 1.0
        
        # 前代法解Ly = e
        y = np.zeros(n)
        for i in range(n):
            y[i] = (e[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
            print(f"y[{i+1}] = {y[i]:.6f}")
        
        # 回代法解Ux = y
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = y[i] - np.dot(U[i, i+1:], x[i+1:])
            print(f"x[{i+1}] = {x[i]:.6f}")
        
        # 將結果放入逆矩陣對應列
        A_inv[:, j] = x
        
        # 驗證每一列
        col_check = np.dot(A, x)
        col_error = np.max(np.abs(col_check - e))
        print(f"第{j+1}列的誤差: {col_error:.6e}")
    
    return A_inv, L, U

# 執行Crout分解和逆矩陣計算
A_inv, L, U = inverse_with_crout(A)

# 顯示L和U矩陣
print("\nL矩陣:")
print(L)
print("\nU矩陣:")
print(U)

# 顯示計算得到的逆矩陣
print("\n計算得到的逆矩陣A^(-1):")
print(A_inv)

# 使用numpy的內建函數計算逆矩陣做比較
np_inv = np.linalg.inv(A)
print("\nNumPy計算的逆矩陣:")
print(np_inv)
np_error = np.max(np.abs(A_inv - np_inv))
print(f"兩種方法的結果誤差: {np_error:.6e}")
