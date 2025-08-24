import numpy as np

def mi_clip_method(matrix):
    """ 米字窗口划分法 """
    assert matrix.shape[0] == matrix.shape[1]
    n = matrix.shape[0]

    positive_diagonal_upper = []  # 主对角线上方
    positive_diagonal_lower = []  # 主对角线下方  
    negative_diagonal_upper = []  # 副对角线上方
    negative_diagonal_lower = []  # 副对角线下方
    upper_matrix = [] # 上矩阵
    lower_matrix = [] # 下矩阵
    left_matrix = [] # 左矩阵
    right_matrix = [] # 右矩阵

    for i in range(n):
        for j in range(n):
            """ 正对角线上 """
            if i <= j:
                positive_diagonal_upper.append(matrix[i][j])
            """ 正对角线下 """
            if i >= j:
                positive_diagonal_lower.append(matrix[i][j])
            """ 副对角线上 """
            if i + j <= n - 1:
                negative_diagonal_upper.append(matrix[i][j])
            """ 副对角线下 """
            if i + j >= n - 1:
                negative_diagonal_lower.append(matrix[i][j])
            """上矩阵"""
            if i <= int(n/2):
                upper_matrix.append(matrix[i][j])
            """下矩阵"""
            if i >= int(n/2):
                lower_matrix.append(matrix[i][j])
            """ 左矩阵 """
            if j <= int(n/2):
                left_matrix.append(matrix[i][j])
            """右矩阵"""
            if j >= int(n/2):
                right_matrix.append(matrix[i][j])
    return [positive_diagonal_upper, positive_diagonal_lower,  negative_diagonal_upper,  negative_diagonal_lower,
            upper_matrix, lower_matrix , left_matrix , right_matrix ]



def tian_clip_method(matrix):
    """ 田字窗口划分法 """
    assert matrix.shape[0] == matrix.shape[1]
    n = matrix.shape[0]

    top_left_matrix = []  # 左上
    top_right_matrix = []  # 右上
    bottom_left_matrix = []  # 左下
    bottom_right_matrix = []  # 右下

    upper_matrix = [] # 上矩阵
    lower_matrix = [] # 下矩阵
    left_matrix = [] # 左矩阵
    right_matrix = [] # 右矩阵

    b = int(n/2)+1

    for i in range(n):
        for j in range(n):
            """ 左上 """
            if i < b and j < b:
                top_left_matrix.append(matrix[i][j])
            """ 左下 """
            if i >= b-1 and j<b:

                bottom_left_matrix.append(matrix[i][j])
            """ 右上 """
            if i < b and j>=b-1:
                top_right_matrix.append(matrix[i][j])
            """ 右下 """
            if i >= b-1 and j>=b-1:
                bottom_right_matrix.append(matrix[i][j])
            """上矩阵"""
            if j >= int(n/4) and j < n-int(n/4) and i < b:
                upper_matrix.append(matrix[i][j])
            """下矩阵"""
            if j >= int(n/4) and j < n-int(n/4) and i >= b-1:
                lower_matrix.append(matrix[i][j])

            """ 左矩阵 """
            if j<b and i>= int(n/4) and i < n-int(n/4):
                left_matrix.append(matrix[i][j])
            """右矩阵"""
            if j>=b-1 and i>= int(n/4) and i < n-int(n/4):
                right_matrix.append(matrix[i][j])
                
    return [top_left_matrix, bottom_left_matrix,  top_right_matrix,  bottom_right_matrix,
            upper_matrix, lower_matrix , left_matrix , right_matrix ]


def hua_clip_method(matrix):
    """ 滑动窗口划分法 """
    assert matrix.shape[0] == matrix.shape[1]
    n = matrix.shape[0]
    windows = []
    window_size = int(n/2) + 1  # 窗口大小为 n+1
    stride = 1
    # 计算窗口可以滑动的次数
    max_i = n - window_size + 1
    max_j = n - window_size + 1
    # 滑动窗口并提取每个窗口
    for i in range(0, max_i, stride):
        for j in range(0, max_j, stride):
            window = matrix[i:i+window_size, j:j+window_size]
            windows.append(window.copy())  # 使用 copy 确保每个窗口是独立的数组
    return windows

def polygon56_clip_method(matrix_):
    def split_matrix_by_diagonals(matrix):
        n = matrix.shape[0]
        diagonals = []
        # 处理主对角线方向（左上 -> 右下）的斜线，k 为偏移量
        for k in range(-(n - 1), n):
            diag_elements = np.diag(matrix, k).tolist()
            diagonals.append(diag_elements)
        return diagonals

    """ 五六边型划分方法 """
    assert matrix_.shape[0] == matrix_.shape[1]
    n = matrix_.shape[0]
    windows = []

    for i in range(4):
        window_5 = []
        window_6 = []
        matrix = np.rot90(matrix_, k=i)
        top_left_matrix = matrix[:int(n/2)+1, :int(n/2)+1]
        diagonals = split_matrix_by_diagonals(top_left_matrix)
        mid = len(diagonals) // 2  # 中间元素索引
        offset = (int(len(diagonals)/2)+1 ) // 2  # 向两侧扩展的偏移量
        start = mid - offset
        end = mid + offset + 1
        for i in diagonals[start:end]:
            window_6.extend(i)

        for i in range(n):
            for j in range(n):
                if (j >= int(n/4) and j < n-int(n/4) and i <= int(n/4)) or (i <= j and i + j <= n - 1 and i > int(n/4)):
                    window_5.append(matrix[i][j])

        windows.append(window_5)
        windows.append(window_6)

    return windows