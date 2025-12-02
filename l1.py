import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def generate_normal_distribution(mu, sigma, n_samples):
    """Генерация случайной величины с нормальным распределением"""
    np.random.seed(42)  # для воспроизводимости результатов
    return np.random.normal(mu, sigma, n_samples)


def calculate_statistics(data, mu, sigma):
    """Вычисление и вывод статистик"""
    sample_mean = np.mean(data)
    sample_std = np.std(data)

    print("=== РЕЗУЛЬТАТЫ ===")
    print(f"Теоретические параметры:")
    print(f"  Математическое ожидание (μ): {mu}")
    print(f"  Стандартное отклонение (σ): {sigma}")
    print(f"\nВыборочные характеристики:")
    print(f"  Выборочное среднее: {sample_mean:.4f}")
    print(f"  Выборочное стандартное отклонение: {sample_std:.4f}")
    print(f"\nРазница между теоретическими и выборочными значениями:")
    print(f"  Δ мат. ожидания: {abs(mu - sample_mean):.4f}")
    print(f"  Δ стандартного отклонения: {abs(sigma - sample_std):.4f}")

    return sample_mean, sample_std


def plot_histogram_with_pdf(data, mu, sigma):
    """Построение гистограммы с теоретической плотностью"""
    plt.figure(figsize=(12, 8))

    # Гистограмма данных
    count, bins, patches = plt.hist(data, bins=50, density=True, alpha=0.7,
                                    color='skyblue', edgecolor='black',
                                    label='Гистограмма данных')

    # Теоретическая плотность распределения
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    theoretical_pdf = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, theoretical_pdf, 'r-', linewidth=2, label='Теоретическая плотность')

    # Добавление линий для математического ожидания и стандартных отклонений
    plt.axvline(mu, color='red', linestyle='--', alpha=0.7, label=f'μ = {mu}')
    for i in range(1, 4):
        plt.axvline(mu + i * sigma, color='orange', linestyle=':', alpha=0.6)
        plt.axvline(mu - i * sigma, color='orange', linestyle=':', alpha=0.6)

    plt.xlabel('Значение')
    plt.ylabel('Плотность вероятности')
    plt.title('Нормальное распределение: гистограмма и теоретическая плотность')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def calculate_sigma_probabilities(data, mu, sigma):
    """Расчет вероятностей попадания в интервалы 1σ, 2σ, 3σ"""
    print("\n=== ВЕРОЯТНОСТИ ПОПАДАНИЯ В ИНТЕРВАЛЫ ===")
    intervals = [1, 2, 3]

    for k in intervals:
        # Теоретическая вероятность
        theoretical_prob = stats.norm.cdf(mu + k * sigma, mu, sigma) - stats.norm.cdf(mu - k * sigma, mu, sigma)

        # Эмпирическая вероятность (из данных)
        empirical_prob = np.sum((data >= mu - k * sigma) & (data <= mu + k * sigma)) / len(data)

        print(f"Интервал {k}σ (±{k}σ):")
        print(f"  Теоретическая вероятность: {theoretical_prob:.4f} ({theoretical_prob * 100:.2f}%)")
        print(f"  Эмпирическая вероятность:  {empirical_prob:.4f} ({empirical_prob * 100:.2f}%)")
        print(f"  Разница: {abs(theoretical_prob - empirical_prob):.4f}")


def plot_parameter_influence():
    """Демонстрация влияния параметров на форму распределения"""
    plt.figure(figsize=(15, 10))

    # Влияние математического ожидания (μ)
    plt.subplot(2, 2, 1)
    x_range = np.linspace(-5, 15, 1000)
    mu_values = [0, 3, 6, 9]
    for mu_val in mu_values:
        pdf = stats.norm.pdf(x_range, mu_val, 2)
        plt.plot(x_range, pdf, label=f'μ = {mu_val}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Влияние математического ожидания (σ = 2)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Влияние стандартного отклонения (σ)
    plt.subplot(2, 2, 2)
    sigma_values = [0.5, 1, 2, 3]
    for sigma_val in sigma_values:
        pdf = stats.norm.pdf(x_range, 5, sigma_val)
        plt.plot(x_range, pdf, label=f'σ = {sigma_val}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Влияние стандартного отклонения (μ = 5)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Сравнение разных распределений
    plt.subplot(2, 2, 3)
    params = [(0, 1), (0, 2), (2, 1), (2, 3)]
    for mu_val, sigma_val in params:
        pdf = stats.norm.pdf(x_range, mu_val, sigma_val)
        plt.plot(x_range, pdf, label=f'μ={mu_val}, σ={sigma_val}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Разные нормальные распределения')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Количественная оценка "правила 3σ"
    plt.subplot(2, 2, 4)
    k_sigma = np.arange(0, 4, 0.1)
    probabilities = [stats.norm.cdf(k, 0, 1) - stats.norm.cdf(-k, 0, 1) for k in k_sigma]
    plt.plot(k_sigma, probabilities, 'b-', linewidth=2)
    plt.axhline(0.6827, color='r', linestyle='--', alpha=0.7, label='68.27% (1σ)')
    plt.axhline(0.9545, color='g', linestyle='--', alpha=0.7, label='95.45% (2σ)')
    plt.axhline(0.9973, color='m', linestyle='--', alpha=0.7, label='99.73% (3σ)')
    plt.xlabel('k (в единицах σ)')
    plt.ylabel('Вероятность')
    plt.title('Вероятность попадания в интервал ±kσ')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Основная функция программы"""
    # Параметры распределения
    mu = 5.0  # математическое ожидание
    sigma = 2.0  # стандартное отклонение
    n_samples = 100000  # количество образцов

    # Генерация данных
    data = generate_normal_distribution(mu, sigma, n_samples)

    # Вычисление статистик
    sample_mean, sample_std = calculate_statistics(data, mu, sigma)

    # Построение гистограммы
    plot_histogram_with_pdf(data, mu, sigma)

    # Расчет вероятностей для интервалов σ
    calculate_sigma_probabilities(data, mu, sigma)

    # Демонстрация влияния параметров
    plot_parameter_influence()

    print("\n=== ВЫВОДЫ ===")
    print("1. Выборочные характеристики близки к теоретическим параметрам")
    print("2. Эмпирические вероятности соответствуют правилу 3σ (68-95-99.7)")
    print("3. Параметр μ определяет центр распределения")
    print("4. Параметр σ определяет ширину и форму распределения")


if __name__ == "__main__":
    main()