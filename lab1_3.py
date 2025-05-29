import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, chi2

from random import random
from math import log

CLT_SUM = 20  # amount of numbers from uniform distribution to get a sum of for CLT


def uniform_distribution(n: int):
    return [random() for _ in range(n)]


# Use CLT
def normal_distribution(n: int, mu: float, sigma: float):
    m = CLT_SUM / 2  # uniform distribution mean
    q = (CLT_SUM / 12)**0.5  # uniform distribution standard deviation
    return [mu + sigma * (sum(uniform_distribution(CLT_SUM)) - m) / q for _ in range(n)]  # N(0,1) -> N(μ,σ^2)


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


def get_confindense_interval_mu(data: list[float], gamma: float, sigma: float = -1):
    if gamma > 1 or gamma < 0:
        return None

    alpha = gamma
    n = len(data)
    avg = sum(data) / n  # average of the data

    if sigma <= 0:
        sigma = (sum([(data[i] - avg)**2 for i in range(n)]) / (n - 1)) ** 0.5  # sigma according to data
        z_l = t.ppf(alpha / 2, n - 1)  # quantile of t-function
        z_r = t.ppf(1 - alpha / 2, n - 1)

    z_l = norm.ppf(alpha / 2)  # left tail
    z_r = norm.ppf(1 - alpha / 2)  # right tail

    mu_l = avg - (z_r * sigma / n**0.5)
    mu_r = avg - (z_l * sigma / n**0.5)

    return (mu_l, mu_r)


def get_confindense_interval_d(data: list[float], gamma: float, mu: float = None):
    if gamma > 1 or gamma < 0:
        return None

    alpha = gamma
    n = len(data)

    if mu is None:
        avg = sum(data) / n  # average of the data
    else:
        avg = mu

    S2 = sum([(data[i] - avg)**2 for i in range(n)]) / (n - 1)  # data variance

    z_l = chi2.ppf(alpha / 2, n - 1)
    z_r = chi2.ppf(1 - alpha / 2, n - 1)

    var_l = (n - 1) * S2 / z_r
    var_r = (n - 1) * S2 / z_l

    return (var_l, var_r)


if __name__ == '__main__':
    MU = 20
    SIGMA = 11
    GAMMA = 0.85

    plt.xlabel('Value')
    plt.ylabel('Density')

    arr500 = normal_distribution(5000, mu=MU, sigma=SIGMA)
    arr50 = normal_distribution(50, mu=MU, sigma=SIGMA)
    # model_histogram_by_bins(arr, 50)
    # plt.show()

    print(f"CONFIDENSE LEVEL mu (known sigma): {get_confindense_interval_mu(arr500, GAMMA, SIGMA)}")  # Check for error
    print(f"CONFIDENSE LEVEL mu (unknown sigma): {get_confindense_interval_mu(arr500, GAMMA)}")  # Check for error
    print(f"CONFIDENSE LEVEL variance (known mu): {get_confindense_interval_d(arr500, GAMMA, MU)}")  # Check for error
    print(f"CONFIDENSE LEVEL variance (unknown mu): {get_confindense_interval_d(arr500, GAMMA)}")  # Check for error
    model_histogram_with_stetgers(arr500)
    plt.title(f'Histogram of 500 Normally Distributed Values')
    plt.show()

    print(f"CONFIDENSE LEVEL mu (known sigma): {get_confindense_interval_mu(arr50, GAMMA, SIGMA)}")
    print(f"CONFIDENSE LEVEL mu (unknown sigma): {get_confindense_interval_mu(arr50, GAMMA)}")  # Check for error
    print(f"CONFIDENSE LEVEL variance (known mu): {get_confindense_interval_d(arr50, GAMMA, MU)}")  # Check for error
    print(f"CONFIDENSE LEVEL variance (unknown mu): {get_confindense_interval_d(arr50, GAMMA)}")  # Check for error
    model_histogram_with_stetgers(arr50)
    plt.title(f'Histogram of 50 Normally Distributed Values')
    plt.show()
