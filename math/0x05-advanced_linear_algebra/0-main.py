#!/usr/bin/env python3

if __name__ == '__main__':
    determinant = __import__('0-determinant').determinant

    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat31 = [[5, 7], [3, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]
    mat7 = [[1, 2, 3, 4], [1, 2, 3, 4], [4, 5, 6, 7], [6, 7, 8, 9]]
    # 0
    mat8 = [[0, 3, 5, 9], [1, 3, 1, 7], [4, 3, 9, 7], [5, 2, 0, 9]]
    # +- 480
    mat9 = [[1, 0, 1], [0, 0, 1], [0, 1, 7]]
    # -1
    mat10 = [[1, 2, 0, 0], [3, 4, 5, 0], [0, 6, 0, 6], [0, 8, 0, 0]]
    # 240

    print(determinant(mat0))
    print(determinant(mat1))
    print(determinant(mat2))
    print(determinant(mat3))
    print(determinant(mat31))
    print(determinant(mat4))
    try:
        determinant(mat5)
    except Exception as e:
        print(e)
    try:
        determinant(mat6)
    except Exception as e:
        print(e)
    print(determinant(mat7))
    print(determinant(mat8))
    print(determinant(mat9))
    print(determinant(mat10))
