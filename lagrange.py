import numpy as np
import matplotlib.pyplot as plt


def calculate_l_value(iteration, points_matrix, x_value):
    l_value = 1
    n = len(points_matrix)
    for j in range(n):
        if j == iteration:
            continue
        xj, yj = points_matrix[j]
        l_value *= float(x_value - xj) / float(points_matrix[iteration][0] - xj)
    return l_value


# points is a 2d array with [x,y] per each record
def lagrange(points_matrix, x_value):
    result = 0
    n = len(points_matrix)
    for i in range(n):
        xi, yi = points_matrix[i]
        result += yi * calculate_l_value(i, points_matrix, x_value)
    return result


def execute_method(input_matrix, method, out_size):
    out_matrix = []
    for i in range(out_size):
        if method == "-l":
            single_record = [i, lagrange(input_matrix, i)]
            out_matrix.append(single_record)
    return out_matrix


def main():
    my_data = [[1, 1], [3, 9], [5, 25], [6, 36], [8, 64], [10, 100], [12, 144]]
    out_data = execute_method(my_data, "-l", 13)

    print(out_data)


if __name__ == "__main__":
    main()
