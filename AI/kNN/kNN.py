#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Tools.scripts.treesync import raw_input
from numpy import *
import operator
import matplotlib.pyplot as plt


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_ilabel = labels[sorted_dist_indicies[i]]
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


group1, labels1 = create_data_set()

print(classify0([0, 0], group1, labels1, 3))


def file2matrix(filename):
    fr = open(filename)
    array_o_lines = fr.readlines()
    number_of_lines = len(array_o_lines)
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_o_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0: 3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def draw_data(data1, data2, data3):
    fg = plt.figure()
    ax = fg.add_subplot(111)
    ax.scatter(data1, data2, 15.0 * array(data3), 15.0 * array(data3))
    plt.show()


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(raw_input("percentage of time spent playing video games?"))
    ff_miles = float(raw_input("frequent flier miles earned per year?"))
    ice_cream = float(raw_input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_array = array([ff_miles, percent_tats, ice_cream])
    classfier_result = classify0((in_array - min_vals) / ranges, norm_mat, dating_labels, 3)
    print("you will probably like this person: " + result_list[classfier_result - 1])
    draw_data(dating_data_mat[:, 0], dating_data_mat[:, 1], dating_labels)


if __name__ == "__main__":
    print("main")
    classify_person()
