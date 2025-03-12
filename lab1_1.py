import numpy as np
import matplotlib.pyplot as plt
from random import random
from math import log

CLT_SUM = 20  # amount of numbers from uniform distribution to get a sum of for CLT


def uniform_distribution(n: int):
    return [random() for _ in range(n)]


# Use CLT
def normal_distribution(n: int):
    return [(sum(uniform_distribution(CLT_SUM)) - CLT_SUM / 2) / (CLT_SUM / 12) ** 0.5 for _ in range(n)]  # normalized


def get_interval_frequencies(data: list[float], num_bins: int) -> list[float]:
    N = len(data)
    max_value = max(data)
    min_value = min(data)
    bin_width = (max_value - min_value) / num_bins

    count_bins = [0] * num_bins

    for x in data:
        bin_ind = int((x - min_value) // bin_width)
        bin_ind -= int(bin_ind == num_bins)  # correct for max_value
        count_bins[bin_ind] += 1

    return [count / (N * bin_width) for count in count_bins]


def get_histogram_edges(data: list[float], num_bins: int):
    max_value = max(data)
    min_value = min(data)
    bin_width = (max_value - min_value) / num_bins
    return [min_value + i * bin_width for i in range(num_bins + 1)]


def model_histogram(x: list[float], y: list[float], width):
    print(x)
    print(y)
    plt.bar(x, y, width=width, align='edge', color='b', edgecolor='black')

# REWRITE THIS SHIT


# def show_histo(data: list[float], num_bins: int):
#     N = len(data)  # number of RV
#     len_bin = (max(data) - min(data)) / num_bins  # length of single bin
#     i = min(data)
#     arr_bins = [[i, i := (i + len_bin)] for _ in range(num_bins)]  # array of intervals [start; end]
#     arr_bins[-1][1] = max(data)  # eliminate error (python can't calculate precisely)
#     # print(arr_bins)

#     # Frequency calculation
#     arr_heights = []
#     bin_ind = 0
#     count = 0
#     data_sorted = sorted(data)
#     i = 0
#     while i < N:
#         if data_sorted[i] > arr_bins[bin_ind][1]:
#             arr_heights.append(count / (N * len_bin))
#             bin_ind += 1
#             count = 0
#             continue
#         count += 1
#         i += 1
#     else:
#         arr_heights.append(count / (N * len_bin))

#     # arr_heights and arr_bins alignment
#     for _ in range(len(arr_bins) - len(arr_heights)):
#         arr_heights.append(0)

#     # print(len(arr_bins))
#     # print(len(arr_heights))
#     # print(data_sorted)
#     print(arr_heights)

#     bin_starts = [b[0] for b in arr_bins]  # x positions
#     bin_widths = [len_bin for b in arr_bins]  # Widths
#     plt.bar(bin_starts, arr_heights, width=bin_widths, align='edge', color='b', edgecolor='black')
#     # plt.show()


def model_histogram_by_bins(data: list[float], num_bins: int):
    # DO SOMETHING WITH THE REPEATING CODE!
    max_value = max(data)
    min_value = min(data)
    bin_width = (max_value - min_value) / num_bins

    freqs = get_interval_frequencies(data, num_bins)
    starts = get_histogram_edges(data, num_bins)
    starts.pop() # omit the most right edge
    model_histogram(starts, freqs, bin_width)


def model_histogram_with_stetgers(data: list[float]):
    num_bins = int(1 + log(len(data), 2))
    model_histogram_by_bins(data, num_bins)


# def show_histo_with_stetgers(data: list[float]):
#     num_bins = int(1 + log(len(data), 2))
#     show_histo(data, num_bins)


if __name__ == '__main__':
    N = 1000

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Histogram of {N} Normally Distributed Values')

    # arr = uniform_distribution(N)
    # show_histo(arr, 10)

    arr = normal_distribution(N)
    model_histogram_by_bins(arr, 50)
    plt.show()

    model_histogram_with_stetgers(arr)
    plt.show()
