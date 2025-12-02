import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ===============================
# 1. Параметры нормального распределения
# ===============================
mu = 4         # математическое ожидание
sigma = 2       # стандартное отклонение

# ===============================
# 2. Генерация выборки
# ===============================
np.random.seed(23)
N = 10000
data = np.random.normal(mu, sigma, N)

# ===============================
# 3. Выборочные характеристики
# ===============================
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)

print("ТЕОРЕТИЧЕСКИЕ ПАРАМЕТРЫ:")
print(f"  μ = {mu}")
print(f"  σ = {sigma}")

print("\nВЫБОРОЧНЫЕ ПАРАМЕТРЫ:")
print(f"  mat. ожидание выборки = {sample_mean:.4f}")
print(f"  стандартное отклонение выборки = {sample_std:.4f}")

# ===============================
# 4. Гистограмма + теоретическая плотность
# ===============================
plt.figure(figsize=(10, 6))
plt.hist(data, bins=40, density=True, color='orange', alpha=0.6, label="Гистограмма выборки")

# теоретическая плотность
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
pdf = norm.pdf(x, mu, sigma)
plt.plot(x, pdf, 'b', linewidth=2, label="Теоретическая плотность")

plt.title("Гистограмма и теоретическая плотность нормального распределения")
plt.xlabel("x")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# 5. Вероятности попадания в интервалы
# ===============================
p1 = norm.cdf(mu + sigma, mu, sigma) - norm.cdf(mu - sigma, mu, sigma)
p2 = norm.cdf(mu + 2*sigma, mu, sigma) - norm.cdf(mu - 2*sigma, mu, sigma)
p3 = norm.cdf(mu + 3*sigma, mu, sigma) - norm.cdf(mu - 3*sigma, mu, sigma)

print("\nВЕРОЯТНОСТИ ПОПАДАНИЯ В ИНТЕРВАЛЫ:")
print(f"  P(|X - μ| < 1σ) = {p1:.6f}")
print(f"  P(|X - μ| < 2σ) = {p2:.6f}")
print(f"  P(|X - μ| < 3σ) = {p3:.6f}")
