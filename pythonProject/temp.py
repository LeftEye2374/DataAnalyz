import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, k, c, e

# Параметры материала
Eg_300K = 1.42  # Ширина запрещенной зоны GaAs при 300K (эВ)
alpha = 5.41e-4  # Коэффициент температурной зависимости (эВ/К)
beta = 204        # Параметр температурной зависимости (К)

# Функция для расчета ширины запрещенной зоны
def bandgap(T):
    return Eg_300K - (alpha * T**2) / (T + beta)

# Функция для расчета подвижности электронов
def mobility(T):
    mu_0 = 8500   # Подвижность при 300K (см²/В·с)
    T_ref = 300    # Референсная температура (К)
    return mu_0 * (T_ref / T)**2.3

# Функция для расчета концентрации носителей
def carrier_concentration(T):
    Nc = 4.7e17 * (T/300)**1.5  # Эффективная плотность состояний (см^-3)
    Nv = 7.0e18 * (T/300)**1.5  # Для валентной зоны
    return np.sqrt(Nc * Nv) * np.exp(-bandgap(T)/(2*k*T/e))

# Диапазон температур
temperatures = np.linspace(100, 800, 100)

# Расчет параметров
Eg_values = [bandgap(T) for T in temperatures]
mu_values = [mobility(T) for T in temperatures]
n_values = [carrier_concentration(T) for T in temperatures]

# Построение графиков
plt.figure(figsize=(12, 8))

# График ширины запрещенной зоны
plt.subplot(2, 2, 1)
plt.plot(temperatures, Eg_values, 'r-', linewidth=2)
plt.title('Зависимость ширины запрещенной зоны от температуры')
plt.xlabel('Температура (K)')
plt.ylabel('Eg (эВ)')
plt.grid(True)

# График подвижности
plt.subplot(2, 2, 2)
plt.plot(temperatures, mu_values, 'b-', linewidth=2)
plt.title('Зависимость подвижности от температуры')
plt.xlabel('Температура (K)')
plt.ylabel('Подвижность (см²/В·с)')
plt.grid(True)

# График концентрации носителей
plt.subplot(2, 2, 3)
plt.semilogy(temperatures, n_values, 'g-', linewidth=2)
plt.title('Зависимость концентрации носителей от температуры')
plt.xlabel('Температура (K)')
plt.ylabel('Концентрация (см^-3)')
plt.grid(True)

plt.tight_layout()
plt.show()