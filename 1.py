import random
import matplotlib.pyplot as plt


def massiv_random(size):
    """Генерирует список случайных чисел с равномерным распределением из интервала [0, 1]."""
    return [random.random() for i in range(size)]


def normal_random(size):
    """Генерирует список случайных чисел с нормальным распределением, используя ЦПТ."""
    normal_numbers = []
    for i in range(size):
        summ = sum(massiv_random(20))  # Суммируем 20 равномерных
        normal_numbers.append(summ)
    return normal_numbers


def frequencies(data, num_intervals):
    """Вычисляет частоты попадания значений в заданные интервалы."""
    min_val = min(data)
    max_val = max(data)
    w = (max_val - min_val) / num_intervals
    f = [0] * num_intervals
    intervals = []

    for i in range(num_intervals):
        lower = min_val + i * w
        upper = min_val + (i + 1) * w
        intervals.append((lower, upper))

        for value in data:
            if lower <= value < upper:
                f[i] += 1
            elif i == num_intervals - 1 and value == max_val:  # Обработка верхнего порога
                f[i] += 1
    return f, intervals


def create(data, num_intervals, title):
    """Строит гистограмму с использованием matplotlib."""
    f, intervals = frequencies(data, num_intervals)

    interval_centers = [(interval[0] + interval[1]) / 2 for interval in intervals]

    plt.figure(figsize=(10, 6))
    plt.bar(interval_centers, f, width=(intervals[0][1] - intervals[0][0]), edgecolor='black')
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.title(title + f" (Интервалов: {num_intervals})")
    plt.grid(True)
    plt.show()


def sturges(data_size):
    """Определяет количество интервалов с помощью формулы Стерджеса."""
    return int(1 + 3.322 * (data_size)**0.1)


# 1. Генерация случайных чисел
uniform_100 = massiv_random(100)
uniform_1000 = massiv_random(1000)

normal_100 = normal_random(100)
normal_1000 = normal_random(1000)


# 2. Построение гистограмм с заданным количеством интервалов
intervals = [5, 10, 50]

for num_intervals in intervals:
    create(uniform_100, num_intervals, "Равномерное распределение (100)")
    create(uniform_1000, num_intervals, "Равномерное распределение (1000)")
    create(normal_100, num_intervals, "Нормальное распределение (100)")
    create(normal_1000, num_intervals, "Нормальное распределение (1000)")


# 3. Построение гистограмм с количеством интервалов по формуле Стерджеса
sturges_100 = sturges(100)
sturges_1000 = sturges(1000)

create(uniform_100, sturges_100, "Равномерное распределение (100) - Стерджес")
create(uniform_1000, sturges_1000, "Равномерное распределение (1000) - Стерджес")
create(normal_100, sturges_100, "Нормальное распределение (100) - Стерджес")
create(normal_1000, sturges_1000, "Нормальное распределение (1000) - Стерджес")

print("Программа выполнена.")
