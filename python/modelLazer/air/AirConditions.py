import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, hbar, epsilon_0, mu_0
from scipy.integrate import odeint
from datetime import datetime

# Нормировочные коэффициенты
E_norm = 1e6  # Нормировка электрического поля [В/м]
z_norm = 1e-3  # Нормировка расстояния [м]

# Параметры лазера на GaAs (нормированные)
wavelength = 940e-9 / z_norm  # Безразмерная длина волны
frequency = c / wavelength
omega = 2 * np.pi * frequency
E0 = 10.0 / E_norm  # Нормированная амплитуда поля
distance = 0.01 / z_norm  # Нормированная длина резонатора
beam_diameter = 0.0005 / z_norm
beam_area = np.pi * (beam_diameter / 2) ** 2
t_max = 10e-9
num_points = 500  # Уменьшим число точек для стабильности

# Параметры материала GaAs (нормированные)
sigma = 1e-6 * z_norm / (mu_0 * c)  # Нормированная проводимость
n0 = 3.5
n2 = 1e-17 * E_norm ** 2  # Нормированный нелинейный коэффициент


def wave_equation(E, z):
    E_real, E_imag = E

    # Малые возмущения для стабильности
    eps = 1e-10
    E_abs_squared = E_real ** 2 + E_imag ** 2 + eps

    # Нелинейный показатель преломления с ограничением
    n_squared = np.minimum((n0 + n2 * E_abs_squared) ** 2, 100)

    # Уравнения в безразмерной форме
    dE_real_dz = -sigma * omega * E_imag / (2 * n0 ** 2) - (n_squared / n0 ** 2 - 1) * omega ** 2 * E_real / (
                2 * c ** 2)
    dE_imag_dz = sigma * omega * E_real / (2 * n0 ** 2) - (n_squared / n0 ** 2 - 1) * omega ** 2 * E_imag / (2 * c ** 2)

    return [dE_real_dz, dE_imag_dz]


# Начальные условия
E_init = [E0, 0]  # Только действительная и мнимая части

# Пространственная сетка
z = np.linspace(0, distance, num_points)

# Решение с использованием odeint
try:
    solution = odeint(wave_equation, E_init, z, rtol=1e-6, atol=1e-8)
    E_real = solution[:, 0] * E_norm
    E_imag = solution[:, 1] * E_norm
    z_phys = z * z_norm

    # Расчет интенсивности
    intensity = 0.5 * c * epsilon_0 * (E_real ** 2 + E_imag ** 2)

    # Построение графиков
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(z_phys * 1e3, E_real, label='Re(E)')
    plt.plot(z_phys * 1e3, E_imag, label='Im(E)')
    plt.title('Электрическое поле в резонаторе')
    plt.xlabel('Положение (мм)')
    plt.ylabel('Поле (В/м)')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.semilogy(z_phys * 1e3, intensity)
    plt.title('Интенсивность излучения')
    plt.xlabel('Положение (мм)')
    plt.ylabel('Интенсивность (Вт/м²)')
    plt.grid()

    plt.tight_layout()

    # Сохранение результатов
    filename = f'gaas_laser_result_{datetime.now().strftime("%H%M%S")}.png'
    plt.savefig(filename, dpi=120)
    plt.close()

    print(f"Моделирование успешно завершено! Результаты сохранены в {filename}")
    print(f"Максимальная интенсивность: {np.max(intensity):.2e} Вт/м²")

except Exception as e:
    print(f"Ошибка при решении: {str(e)}")
    print("Попробуйте уменьшить параметры n2 или E0")