import random
import matplotlib.pyplot as plt


def inverse(size):
    """Генерирует выборку из распределения с f(x) = 2x на (0, 1) методом обратной функции."""
    us = [random.random() for i in range(size)]
    ts = [x**0.5 for x in us]  # Обратная функция: F^-1(x) = sqrt(x)
    return ts


def frequencies(data, num_intervals):
    """Вычисляет частоты попадания значений в интервалы."""
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
        for i in range(num_intervals):
            if intervals[i][0] <= value < intervals[i][1]:
                f[i] += 1
                break

    return f, intervals


def create(data, num_intervals, title):
    """Создает и отображает гистограмму с наложенной функцией плотности."""
    f, intervals = frequencies(data, num_intervals)
    interval_centers = [(interval[0] + interval[1]) / 2 for interval in intervals]

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=num_intervals, density=True, edgecolor='black')  # Построение гистограммы
    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.title(title + f" (Интервалов: {num_intervals})")
    plt.grid(True)

    # Наложение функции плотности
    import numpy as np  # Импортируем для linspace
    x = np.linspace(0, 1, 100)  # создаем массив точек для графика функции
    pdf = 2 * x  # Функция плотности f(x) = 2x
    plt.plot(x, pdf, 'r-', linewidth=2, label='f(x) = 2x')
    plt.legend()

    plt.show()


def sturges(data_size):
    """Вычисляет количество интервалов по формуле Стерджеса."""
    return int(1 + 3.322 * (data_size**0.1))  # приближение логарифма


def statistics(data):
    """Вычисляет основные характеристики выборки."""
    n = len(data)
    mean = sum(data) / n

    # Мода (в данном случае - приближенно, как центр наиболее частого интервала)
    f, intervals = frequencies(data, sturges(n))
    max_f_i = f.index(max(f))
    mode = (intervals[max_f_i][0] + intervals[max_f_i][1]) / 2

    # Медиана
    s_data = sorted(data)
    if n % 2 == 0:
        median = (s_data[n // 2 - 1] + s_data[n // 2]) / 2
    else:
        median = s_data[n // 2]

    # Выборочная дисперсия
    v_sample = sum([(x - mean)**2 for x in data]) / n

    # Исправленная дисперсия
    v_corrected = sum([(x - mean)**2 for x in data]) / (n - 1)

    # Среднеквадратическое отклонение
    std_dev = v_sample**0.5

    # Эксцесс (только выборочный, т.к. в библиотеках обычно тоже выборочный)
    sum_4 = sum([(x - mean)**4 for x in data]) / n
    kurt = (sum_4 / (std_dev**4)) - 3

    # Коэффициент асимметрии (только выборочный)
    sum_3 = sum([(x - mean)**3 for x in data]) / n
    skew = sum_3 / (std_dev**3)

    return {
        "Среднее значение": mean,
        "Мода": mode,
        "Медиана": median,
        "Выборочная дисперсия": v_sample,
        "Исправленная дисперсия": v_corrected,
        "Среднеквадратическое отклонение": std_dev,
        "Эксцесс": kurt,
        "Коэффициент асимметрии": skew,
    }


def statistics_lib(data):
    """Вычисляет характеристики выборки с использованием библиотечных функций (приближенно)."""

    import numpy as np  # Теперь импортируем только здесь, т.к. matplotlib уже импортирован
    from scipy.stats import skew, kurtosis  # импортируем статистические функции

    data_np = np.array(data)  # Преобразуем в numpy массив

    return{
        "Среднее значение": np.mean(data_np),
        "Медиана": np.median(data_np),
        "Дисперсия": np.var(data_np, ddof=0),
        "Исправленная дисперсия": np.var(data_np, ddof=1),
        "Среднеквадратическое отклонение": np.std(data_np, ddof=0),
        "Коэффициент асимметрии": skew(data_np),
        "Эксцесс": kurtosis(data_np)
    }


# 1. Получение данных
viborka_sizes = [50, 500, 1000]
for size in viborka_sizes:
    data = inverse(size)

    # 2. Гистограмма
    num_intervals = sturges(size)
    create(data, num_intervals, f"Выборка (размер={size}), метод обратной функции")

    # 4. Характеристики выборки
    stats_calculated = statistics(data)
    stats_lib = statistics_lib(data)

    print(f"\nРазмер выборки: {size}")
    print("Вычисленные характеристики:")
    for key, value in stats_calculated.items():
        print(f"{key}: {value}")

    print("\nХарактеристики с использованием библиотечных функций:")
    for key, value in stats_lib.items():
        print(f"{key}: {value}")
