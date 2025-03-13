import numpy as np
import matplotlib.pyplot as plt
from random import random
from math import log

CLT_SUM = 20  # amount of numbers from uniform distribution to get a sum of for CLT


def uniform_distribution(n: int):
    return [random() for _ in range(n)]


# Use CLT
def normal_distribution(n: int):
    m = CLT_SUM / 2  # uniform distribution mean
    q = CLT_SUM / 12  # uniform distribution standard deviation
    return [(sum(uniform_distribution(CLT_SUM)) - m) / q for _ in range(n)]  # normalized


def get_interval_frequencies(data: list[float], num_bins: int) -> list[float]:
    N = len(data)
    max_value = max(data)
    min_value = min(data)
    bin_width = (max_value - min_value) / num_bins

    count_bins = [0] * num_bins

    for x in data:
        bin_ind = int((x - min_value) // bin_width)
        bin_ind -= int(bin_ind == num_bins)  # correct error for max_value
        count_bins[bin_ind] += 1

    freqs = [count / (N * bin_width) for count in count_bins]
    print(f"SUM OF BINS: {sum([freq*bin_width for freq in freqs])}")

    return [count / (N * bin_width) for count in count_bins]


def get_histogram_edges(data: list[float], num_bins: int):
    max_value = max(data)
    min_value = min(data)
    bin_width = (max_value - min_value) / num_bins
    return [min_value + i * bin_width for i in range(num_bins + 1)]


def model_histogram(x: list[float], y: list[float], width):
    plt.bar(x, y, width=width, align='edge', color='b', edgecolor='black')


def model_histogram_by_bins(data: list[float], num_bins: int):
    max_value = max(data)
    min_value = min(data)
    bin_width = (max_value - min_value) / num_bins

    freqs = get_interval_frequencies(data, num_bins)
    starts = get_histogram_edges(data, num_bins)
    starts.pop()  # omit the most right edge
    model_histogram(starts, freqs, bin_width)


def model_histogram_with_stetgers(data: list[float]):
    num_bins = int(1 + log(len(data), 2))
    model_histogram_by_bins(data, num_bins)


if __name__ == '__main__':
    N = 1000

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Histogram of {N} Uniform Distributed Values')

    arr = uniform_distribution(N)
    model_histogram_by_bins(arr, 10)
    plt.show()

    plt.title(f'Histogram of {N} Normally Distributed Values')

    arr = normal_distribution(N)
    model_histogram_by_bins(arr, 50)
    plt.show()

    model_histogram_with_stetgers(arr)
    plt.show()
