import numpy as np
import matplotlib.pyplot as plt
from random import random
import statistics
import scipy

from math import pi as PI
from math import e as E
from math import log

from lab1_1 import model_histogram_with_stetgers, get_histogram_edges, get_interval_frequencies


def nd_func(x: float, m: float, q: float):
    coef = 1 / (q * (2 * PI) ** 0.5)
    power = -((x - m) ** 2) / (2 * q ** 2)
    return coef * E ** power


def get_random_values_by_truncation(n: int, x_min, x_max, func):
    data = []
    while len(data) < n:
        x = random() * (x_max - x_min) + x_min  # random number in [x_min; x_max]
        y = random() * (func(m) - func(x_min)) + func(x_min)  # random number in [y_min; y_max]
        if func(x) >= y:
            data.append(x)

    return data


def get_mode(x: list[float]):
    if not x:
        return 0

    counter = dict()
    for el in x:
        if el not in counter:
            counter[el] = 0
        counter[el] += 1

    max_count = 0
    mode = 0
    for el in x:
        if counter[el] > max_count:
            max_count = counter[el]
            mode = el

    return mode


def get_median(x: list[float]):
    if not x:
        return 0

    middle = len(x) // 2

    if len(x) % 2 == 0:
        return (x[middle] + x[middle - 1]) / 2

    return x[middle]


def get_raw_moment(x: list[float], order=1):
    return sum([k**order for k in x]) / len(x)


def get_moment(x: list[float], order=1):
    variance = 0

    m = get_raw_moment(x)
    n = len(x)

    for el in x:
        variance += (el - m)**order

    variance /= n

    return variance


def get_unbiased_variance(x: list[float]):
    n = len(x)
    return get_moment(x, 2) * n / (n - 1)


def get_standard_deviation(x: list[float]):
    return get_unbiased_variance(x)**0.5


def get_asymmetry(x: list[float]):
    nu_3 = get_moment(x, 3)
    q = get_standard_deviation(x)
    return nu_3 / q**3


def get_excess(x: list[float]):
    nu_4 = get_moment(x, 4)
    q = get_standard_deviation(x)
    return nu_4 / q**4 - 3


if __name__ == '__main__':
    n = 10000
    m = 0
    q = 1
    k = 3

    data_user = get_random_values_by_truncation(n, m - q * k, m + q * k, lambda x: nd_func(x, m, q))
    data_50 = get_random_values_by_truncation(50, m - q * k, m + q * k, lambda x: nd_func(x, m, q))
    data_500 = get_random_values_by_truncation(500, m - q * k, m + q * k, lambda x: nd_func(x, m, q))
    data_1000 = get_random_values_by_truncation(1000, m - q * k, m + q * k, lambda x: nd_func(x, m, q))
    # print(sorted([int(x * 100) / 100 for x in data]))
    model_histogram_with_stetgers(data_user)

    x = np.arange(m - k * q, m + k * q, 0.02)
    y = np.array([nd_func(i, m, q) for i in x])
    plt.plot(x, y, "r")

    num_bins = int(1 + log(len(data_user), 2))
    freqs = get_interval_frequencies(data_user, num_bins)  # that's our y
    edges = get_histogram_edges(data_user, num_bins)  # that's our x
    middles = [(edges[i] + edges[i - 1]) / 2 for i in range(1, len(edges))]  # calculate middle of each bin

    data_discret = []
    for i in range(num_bins):
        data_discret += [middles[i]] * int(freqs[i] * 100)

    my_mode_idiot = get_mode(data_user)
    my_mode = get_mode(data_discret)
    my_median = get_median(data_discret)
    my_mean = get_raw_moment(data_discret, 1)
    my_variance = get_moment(data_discret, 2)
    my_unbiased_variance = get_unbiased_variance(data_discret)
    my_standard_deviation = get_standard_deviation(data_discret)
    my_excess = get_excess(data_discret)
    my_asymmetry = get_asymmetry(data_discret)

    mode_idiot = statistics.mode(data_user)  # if not discret than the function will just find the most repeated value like an idiot
    mode = statistics.mode(data_discret)
    median = statistics.median(data_discret)
    mean = statistics.mean(data_discret)
    variance = statistics.pvariance(data_discret)
    unbiased_variance = statistics.variance(data_discret)
    standard_deviation = statistics.stdev(data_discret)
    excess = scipy.stats.kurtosis(data_discret)  # Fisher definition is used. (3rd semester's flashbacks...)
    asymmetry = scipy.stats.skew(data_discret)

    print(f"My data:")
    print(f"Mode_idiot: {my_mode_idiot}")
    print(f"Mode: {my_mode}")
    print(f"Median: {my_median}")
    print(f"Mean: {my_mean}")
    print(f"Variance: {my_variance}")
    print(f"Unbiased variance: {my_unbiased_variance}")
    print(f"Standard deviation: {my_standard_deviation}")
    print(f"Excess (kurtosis): {my_excess}")
    print(f"Asymmetry (skew): {my_asymmetry}")

    print()

    print("Library methods:")
    print(f"Mode_idiot: {mode_idiot}")
    print(f"Mode: {mode}")
    print(f"Median: {median}")
    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print(f"Unbiased variance: {unbiased_variance}")
    print(f"Standard deviation: {standard_deviation}")
    print(f"Excess (kurtosis): {excess}")
    print(f"Asymmetry (skew): {asymmetry}")

    plt.show()
