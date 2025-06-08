import numpy as np

def mi_clip_method(matrix):
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

