import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants

# Константы
eV_to_J = physical_constants['electron volt-joule relationship'][0]

# Параметры моделирования
pressures = np.linspace(0.05, 1.0, 100)  # Диапазон давлений (атм)
v3_ratios = np.linspace(10, 200, 100)    # Диапазон соотношений V/III

# Функция для расчета содержания Al в зависимости от давления и соотношения потоков
def al_content(p, v3):
    return 0.3 + 0.2 * np.exp(-p/0.3) + 0.1 * np.tanh(v3/100 - 0.5)

# Функция для расчета подвижности в зависимости от параметров роста
def mobility(p, v3):
    return 5000 * (1 - 0.5 * np.exp(-p/0.2)) * (1 - 0.3 * np.exp(-(v3-50)**2/2000))

# Функция для расчета концентрации носителей
def carrier_conc(p, v3):
    return 1e16 * (1 + 10 * p) * np.exp(-(v3-100)**2/5000)

# Создание сетки для расчетов
P, V3 = np.meshgrid(pressures, v3_ratios)
X_Al = al_content(P, V3)
Mu = mobility(P, V3)
N = carrier_conc(P, V3)

# Построение графиков
plt.figure(figsize=(18, 5))

# График содержания алюминия
plt.subplot(1, 3, 1)
contour = plt.contourf(P, V3, X_Al, levels=20, cmap='viridis')
plt.colorbar(contour, label='Содержание Al (x в AlxGa1-xAs)')
plt.title('Зависимость содержания Al от давления и V/III')
plt.xlabel('Давление (атм)')
plt.ylabel('Соотношение V/III')
plt.grid(True)

# График подвижности
plt.subplot(1, 3, 2)
contour = plt.contourf(P, V3, Mu, levels=20, cmap='plasma')
plt.colorbar(contour, label='Подвижность (см²/В·с)')
plt.title('Зависимость подвижности от давления и V/III')
plt.xlabel('Давление (атм)')
plt.ylabel('Соотношение V/III')
plt.grid(True)

# График концентрации носителей
plt.subplot(1, 3, 3)
contour = plt.contourf(P, V3, N, levels=20, cmap='inferno')
plt.colorbar(contour, label='Концентрация (см^-3)')
plt.title('Зависимость концентрации от давления и V/III')
plt.xlabel('Давление (атм)')
plt.ylabel('Соотношение V/III')
plt.grid(True)

plt.tight_layout()
plt.show()