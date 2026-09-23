import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Параметры моделирования
deposition_rates = np.linspace(0.05, 3.0, 100)  # Диапазон скоростей осаждения (нм/с)
growth_times = np.array([10, 30, 60, 120, 300])  # Времена роста (с)

# Экспериментальные данные для калибровки модели
exp_rates = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
exp_roughness = np.array([0.2, 0.3, 0.5, 1.2, 2.5])  # Шероховатость (нм)
exp_mobility = np.array([8500, 8000, 7500, 6000, 4000])  # Подвижность (см²/В·с)

# Интерполяция экспериментальных данных
roughness_func = interp1d(exp_rates, exp_roughness, kind='cubic', fill_value='extrapolate')
mobility_func = interp1d(exp_rates, exp_mobility, kind='cubic', fill_value='extrapolate')

# Функция для расчета толщины слоя
def layer_thickness(rate, time):
    return rate * time

# Функция для расчета дефектности
def defect_density(rate):
    return 1e4 * (1 + 10 * np.exp((rate - 1.5)/0.5))

# Расчет параметров
roughness_values = roughness_func(deposition_rates)
mobility_values = mobility_func(deposition_rates)
defect_values = defect_density(deposition_rates)

# Построение графиков
plt.figure(figsize=(15, 10))

# График зависимости шероховатости от скорости осаждения
plt.subplot(2, 2, 1)
plt.plot(deposition_rates, roughness_values, 'b-', linewidth=2)
plt.title('Зависимость шероховатости от скорости осаждения')
plt.xlabel('Скорость осаждения (нм/с)')
plt.ylabel('Шероховатость (нм)')
plt.grid(True)

# График зависимости подвижности от скорости осаждения
plt.subplot(2, 2, 2)
plt.plot(deposition_rates, mobility_values, 'r-', linewidth=2)
plt.title('Зависимость подвижности от скорости осаждения')
plt.xlabel('Скорость осаждения (нм/с)')
plt.ylabel('Подвижность (см²/В·с)')
plt.grid(True)

# График зависимости дефектности от скорости осаждения
plt.subplot(2, 2, 3)
plt.semilogy(deposition_rates, defect_values, 'g-', linewidth=2)
plt.title('Зависимость плотности дефектов от скорости осаждения')
plt.xlabel('Скорость осаждения (нм/с)')
plt.ylabel('Плотность дефектов (см⁻²)')
plt.grid(True)

# График зависимости толщины от времени для разных скоростей
plt.subplot(2, 2, 4)
for rate in [0.2, 0.5, 1.0, 2.0]:
    thickness = [layer_thickness(rate, t) for t in growth_times]
    plt.plot(growth_times, thickness, label=f'{rate} нм/с')
plt.title('Зависимость толщины слоя от времени роста')
plt.xlabel('Время роста (с)')
plt.ylabel('Толщина слоя (нм)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()