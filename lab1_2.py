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


# Rejection sampling
def get_random_values_by_truncation(n: int, x_min, x_max, func):
    data = []
    while len(data) < n:
        x = random() * (x_max - x_min) + x_min  # random number in [x_min; x_max]
        y = random() * (func(m) - func(x_min)) + func(x_min)  # random number in [y_min; y_max]
        if func(x) >= y:
            data.append(x)

    return data

###############################
# FUNCTION FOR DISCRET VALUES #
###############################


def get_mode_discret(x: list[float]):
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


def get_median_discret(x: list[float]):
    if not x:
        return 0

    x = sorted(x)

    middle = len(x) // 2

    if len(x) % 2 == 0:
        return (x[middle] + x[middle - 1]) / 2

    return x[middle]


def get_raw_moment_discret(x: list[float], order=1):
    return sum([k**order for k in x]) / len(x)


def get_moment_discret(x: list[float], order=1):
    variance = 0

    m = get_raw_moment_discret(x)
    n = len(x)

    for el in x:
        variance += (el - m)**order

    variance /= n

    return variance


def get_unbiased_variance_discret(x: list[float]):
    n = len(x)
    return get_moment_discret(x, 2) * n / (n - 1)


def get_standard_deviation_discret(x: list[float]):
    return get_unbiased_variance_discret(x)**0.5


def get_asymmetry_discret(x: list[float]):
    nu_3 = get_moment_discret(x, 3)
    q = get_standard_deviation_discret(x)
    return nu_3 / q**3


def get_excess_discret(x: list[float]):
    nu_4 = get_moment_discret(x, 4)
    q = get_standard_deviation_discret(x)
    return nu_4 / q**4 - 3

###############################
# FUNCTIONS FOR INTERVAL DATA # (note that we assume that intervals are the same width)
###############################


def get_mode_interval(x: list[float], y: list[float]):
    if not x or len(x) != len(y):
        return 0

    n = len(x)
    bin_width = x[1] - x[0]  # width of an interval

    yM = 0
    iM = 0  # mode bin index
    for i in range(n):
        if yM < y[i]:
            yM = y[i]
            iM = i

    y_curr = y[iM]
    y_next = y[iM + 1]
    y_prev = y[iM - 1]

    # We find the tallest bin and to find the mode point we intersect diagonals of a trapezium
    # made from the higher points of the tallest bin and the left and right higher points of previous and next bin respectively
    # thus the mode point will be drawn towards the taller neighboring bin
    mode = x[iM] + bin_width * ((y_curr - y_prev) / ((y_curr - y_prev) + (y_curr - y_next)))
    return mode


def get_median_interval(x: list[float], y: list[float]):
    # IDEA: we should traverse through X values until we reach the cumulative sum of
    # bins' areas of more than 0.5. Then we find the Xm such that its rectangle adds up to exactly 0.5

    if not x or len(x) != len(y):
        return 0

    n = len(x)
    bin_width = x[1] - x[0]  # width of an interval

    area = 0
    i = 0
    # Until the area hits 0.5 we accumulate bins' areas
    while i < n:
        if area + bin_width * y[i] >= 0.5:
            break
        area += bin_width * y[i]
        i += 1

    median = (0.5 - area) / y[i] + x[i]
    return median


def get_raw_moment_interval(x: list[float], y: list[float], order=1):
    n = len(x)
    bin_width = x[1] - x[0]
    x_middles = [k + bin_width / 2 for k in x]  # middles of each bins

    y_sum = sum(y)
    y_normalized = [y[i] / y_sum for i in range(n)]
    return sum([y_normalized[i] * x_middles[i]**order for i in range(n)])


def get_moment_interval(x: list[float], y: list[float], order=1):
    n = len(x)
    bin_width = x[1] - x[0]
    x_middles = [k + bin_width / 2 for k in x]

    m = get_raw_moment_interval(x, y, 1)

    y_sum = sum(y)
    y_normalized = [y[i] / y_sum for i in range(n)]  # Don't we need the sum of all probabilities to be equal 1? Well, here we make it real.

    variance = sum([y_normalized[i] * (x_middles[i] - m)**order for i in range(n)])
    return variance


def get_unbiased_variance_interval(x: list[float], y: list[float], n: int):
    return get_moment_interval(x, y, 2) * n / (n - 1)


# Following functions use UNBIASED variance so they need to know the total number of random values.
def get_standard_deviation_interval(x: list[float], y: list[float], n: int):
    return get_unbiased_variance_interval(x, y, n)**0.5


def get_asymmetry_interval(x: list[float], y: list[float], n: int):
    nu_3 = get_moment_interval(x, y, 3)
    q = get_standard_deviation_interval(x, y, n)
    return nu_3 / q**3


def get_excess_interval(x: list[float], y: list[float], n: int):
    nu_4 = get_moment_interval(x, y, 4)
    q = get_standard_deviation_interval(x, y, n)
    return nu_4 / q**4 - 3


if __name__ == '__main__':
    n = 10000
    m = 0
    q = 1
    k = 3

    data_user = get_random_values_by_truncation(n, m - q * k, m + q * k, lambda x: nd_func(x, m, q))
    # data_50 = get_random_values_by_truncation(50, m - q * k, m + q * k, lambda x: nd_func(x, m, q))
    # data_500 = get_random_values_by_truncation(500, m - q * k, m + q * k, lambda x: nd_func(x, m, q))
    # data_1000 = get_random_values_by_truncation(1000, m - q * k, m + q * k, lambda x: nd_func(x, m, q))
    model_histogram_with_stetgers(data_user)

    x = np.arange(m - k * q, m + k * q, 0.02)
    y = np.array([nd_func(i, m, q) for i in x])
    plt.plot(x, y, "r")

    num_bins = int(1 + log(len(data_user), 2))
    freqs = get_interval_frequencies(data_user, num_bins)  # that's our y
    edges = get_histogram_edges(data_user, num_bins)  # that's our x
    middles = [(edges[i] + edges[i - 1]) / 2 for i in range(1, len(edges))]  # calculate middle of each bin

    freq_sum = sum(freqs)
    freq_normalized = [freqs[i] / freq_sum for i in range(len(freqs))]

    data_discret = []
    bin_width = (max(data_user) - min(data_user)) / n
    for i in range(num_bins):
        data_discret += [middles[i]] * int(freq_normalized[i] * 1000)

    my_mode_idiot_discret = get_mode_discret(data_user)
    my_mode_discret = get_mode_discret(data_discret)
    my_median_discret = get_median_discret(data_discret)
    my_mean_discret = get_raw_moment_discret(data_discret, 1)
    my_variance_discret = get_moment_discret(data_discret, 2)
    my_unbiased_variance_discret = get_unbiased_variance_discret(data_discret)
    my_standard_deviation_discret = get_standard_deviation_discret(data_discret)
    my_excess_discret = get_excess_discret(data_discret)
    my_asymmetry_discret = get_asymmetry_discret(data_discret)

    edges.pop()  # remove last edge. That's our x
    my_mode_interval = get_mode_interval(edges, freqs)
    my_median_interval = get_median_interval(edges, freqs)
    my_mean_interval = get_raw_moment_interval(edges, freqs, 1)
    my_variance_interval = get_moment_interval(edges, freqs, 2)
    my_unbiased_variance_interval = get_unbiased_variance_interval(edges, freqs, n)  # additionally we pass number of elements in all intervals
    my_standard_deviation_interval = get_standard_deviation_interval(edges, freqs, n)
    my_excess_interval = get_excess_interval(edges, freqs, n)
    my_asymmetry_interval = get_asymmetry_interval(edges, freqs, n)

    mode_idiot = statistics.mode(data_user)  # if not discret than the function will just find the most repeated value like an idiot
    mode = statistics.mode(data_discret)
    median = statistics.median(data_discret)
    mean = statistics.mean(data_discret)
    variance = statistics.pvariance(data_discret)
    unbiased_variance = statistics.variance(data_discret)
    standard_deviation = statistics.stdev(data_discret)
    excess = scipy.stats.kurtosis(data_discret)  # Fisher definition is used. (3rd semester's flashbacks...)
    asymmetry = scipy.stats.skew(data_discret)

    print(f"My data (discret):")
    print(f"Mode_idiot: {my_mode_idiot_discret}")
    print(f"Mode: {my_mode_discret}")
    print(f"Median: {my_median_discret}")
    print(f"Mean: {my_mean_discret}")
    print(f"Variance: {my_variance_discret}")
    print(f"Unbiased variance: {my_unbiased_variance_discret}")
    print(f"Standard deviation: {my_standard_deviation_discret}")
    print(f"Excess (kurtosis): {my_excess_discret}")
    print(f"Asymmetry (skew): {my_asymmetry_discret}")

    print()

    print(f"My data (interval):")
    print(f"Mode: {my_mode_interval}")
    print(f"Median: {my_median_interval}")
    print(f"Mean: {my_mean_interval}")
    print(f"Variance: {my_variance_interval}")
    print(f"Unbiased variance: {my_unbiased_variance_interval}")
    print(f"Standard deviation: {my_standard_deviation_interval}")
    print(f"Excess (kurtosis): {my_excess_interval}")
    print(f"Asymmetry (skew): {my_asymmetry_interval}")

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
