import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, hbar, epsilon_0, mu_0
from datetime import datetime
from scipy.integrate import odeint

# Параметры модели
wavelength = 940e-9  # Длина волны [м]
frequency = c / wavelength
omega = 2 * np.pi * frequency
E0 = 10.0  # Амплитуда поля [В/м]
distance = 1000.0  # Расстояние [м]
beam_diameter = 0.002  # Диаметр пучка [м]
beam_area = np.pi * (beam_diameter / 2) ** 2  # Площадь пучка [м²]
t_max = 10e-9  # Время моделирования [с]
num_points = 500  # Число точек расчета

# Параметры осадков
precipitation_type = 'rain'  # 'rain' или 'snow'
intensity_level = 'moderate'  # 'light', 'moderate', 'heavy'

# Параметры нелинейной среды
if precipitation_type == 'rain':
    if intensity_level == 'light':
        sigma = 0.01  # Проводимость [См/м]
        n0 = 1.0003  # Линейный показатель преломления
        n2 = 1e-20  # Нелинейный коэффициент [м²/В²]
    elif intensity_level == 'moderate':
        sigma = 0.05
        n0 = 1.0005
        n2 = 1e-19
    else:
        sigma = 0.1
        n0 = 1.0008
        n2 = 1e-18
else:  # snow
    if intensity_level == 'light':
        sigma = 0.005
        n0 = 1.0002
        n2 = 1e-21
    elif intensity_level == 'moderate':
        sigma = 0.02
        n0 = 1.0003
        n2 = 1e-20
    else:
        sigma = 0.05
        n0 = 1.0005
        n2 = 1e-19


# Функция для решения нелинейного волнового уравнения
def solve_wave_equation(z, E):
    E_abs = np.abs(E[0] + 1j * E[1])
    n_squared = (n0 + n2 * E_abs ** 2) ** 2

    dEx_dz = E[2]
    dEy_dz = E[3]

    # Уравнение для второй производной (упрощенное решение)
    term1 = mu_0 * sigma * omega * (E[1])  # Затухание
    term2 = mu_0 * epsilon_0 * n_squared * omega ** 2 * (E[0])  # Дисперсия

    d2Ex_dz2 = term1 + term2
    d2Ey_dz2 = -term1 + term2

    return [dEx_dz, dEy_dz, d2Ex_dz2, d2Ey_dz2]


# Начальные условия (линейная поляризация по x)
E_init = [E0, 0, 0, 0]

# Пространственная сетка
z = np.linspace(0, distance, num_points)

# Численное решение
solution = odeint(solve_wave_equation, E_init, z)
Ex = solution[:, 0]
Ey = solution[:, 1]

# Расчет интенсивности
E_field = Ex + 1j * Ey
intensity = 0.5 * c * epsilon_0 * np.abs(E_field) ** 2

# Временная зависимость интенсивности у лазера (z=0)
t = np.linspace(0, t_max, 100)
I0_t = 0.5 * c * epsilon_0 * E0 ** 2 * np.ones_like(t)

# Энергетические расчеты
photon_energy_ev = (hbar * omega) / e  # Энергия фотона [эВ]
I0 = intensity[0]  # Начальная интенсивность
I_receiver = intensity[-1]  # Интенсивность на приемнике

# Мощности [Вт]
P_initial = I0 * beam_area
P_receiver = I_receiver * beam_area

# Энергии [эВ]
E_initial = (P_initial * t_max) / e
E_receiver = (P_receiver * t_max) / e

# Коэффициенты
transmission = P_receiver / P_initial
losses = 1 - transmission

# Создание фигуры с графиками
plt.figure(figsize=(12, 8))

# 1. Временная зависимость интенсивности у лазера
plt.subplot(2, 2, 1)
plt.plot(t * 1e9, I0_t)
plt.title('Интенсивность у лазера')
plt.xlabel('Время (нс)')
plt.ylabel('Интенсивность (Вт/м²)')
plt.grid()

# 2. Пространственное распределение интенсивности
plt.subplot(2, 2, 2)
plt.plot(z, intensity)
plt.title('Затухание интенсивности (нелинейная модель)')
plt.xlabel('Расстояние (м)')
plt.ylabel('Интенсивность (Вт/м²)')
plt.grid()

# 3. Логарифмический масштаб для затухания
plt.subplot(2, 2, 3)
plt.semilogy(z, intensity)
plt.title('Затухание (лог. шкала)')
plt.xlabel('Расстояние (м)')
plt.ylabel('Интенсивность (Вт/м²)')
plt.grid()

# 4. Энергетический баланс
plt.subplot(2, 2, 4)
plt.text(0.1, 0.4,
         f"Энергетический баланс:\n"
         f"Тип осадков: {precipitation_type}\n"
         f"Интенсивность: {intensity_level}\n"
         f"Излучено: {E_initial / photon_energy_ev:.2e} эВ\n"
         f"Принято: {E_receiver / photon_energy_ev:.2e} эВ\n"
         f"Потери: {losses * 100:.1f}%\n"
         f"Нелинейный коэффициент: n₂ = {n2:.1e} м²/В²",
         bbox=dict(facecolor='white', alpha=0.8))
plt.axis('off')

plt.tight_layout(pad=2.0)

# Сохранение результатов
filename = f'laser_nonlinear_{precipitation_type}_{intensity_level}.png'
plt.savefig(filename, dpi=100, bbox_inches='tight')
plt.close()

# Вывод информации
print(f"Моделирование завершено в {datetime.now().strftime('%H:%M:%S')}")
print(f"Результаты сохранены в файл: {filename}")
print(f"Параметры среды: σ = {sigma:.4e} См/м, n₀ = {n0:.6f}, n₂ = {n2:.1e} м²/В²")
print(f"Начальная интенсивность: {I0:.4e} Вт/м²")
print(f"Интенсивность на приемнике: {I_receiver:.4e} Вт/м²")