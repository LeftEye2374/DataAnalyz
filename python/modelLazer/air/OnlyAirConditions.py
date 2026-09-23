import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, hbar, epsilon_0, mu_0
from datetime import datetime

## Параметры модели
wavelength = 940e-9  # Длина волны (м)
frequency = c / wavelength
omega = 2 * np.pi * frequency
E0 = 10.0  # Амплитуда поля (В/м)

# Параметры воздушной среды
n_air = 1.000293  # Показатель преломления воздуха
sigma_air = 5e-15  # Проводимость воздуха (См/м)

# Геометрия системы
distance = 1000.0  # Расстояние до приемника (м)
beam_diameter = 0.002  # Диаметр пучка (м)
beam_area = np.pi * (beam_diameter / 2) ** 2

# Временные параметры
t_max = 10e-9  # Время моделирования (с)
num_t_points = 1000
t = np.linspace(0, t_max, num_t_points)

# Пространственные параметры
num_z_points = 1000
z = np.linspace(0, distance, num_z_points)


## Модель распространения волны
def wave_propagation(z, t):
    # Коэффициент затухания
    alpha = (mu_0 * sigma_air * c ** 2) / (2 * n_air)
    # Волновое число
    k = (omega * n_air) / c

    # Учет затухания и фазового набега
    return E0 * np.exp(-alpha * z) * np.cos(omega * t - k * z)


## Расчет интенсивности
def intensity(z, t):
    return 0.5 * c * epsilon_0 * n_air * wave_propagation(z, t) ** 2


## Энергетические расчеты
photon_energy_ev = (hbar * omega) / e


def calculate_energies():
    I0 = intensity(0, 0)
    P_initial = I0 * beam_area

    # Коэффициент пропускания
    T = np.exp(-2 * (mu_0 * sigma_air * c ** 2) / (2 * n_air) * distance)
    P_receiver = P_initial * T

    E_initial = P_initial * t_max
    E_receiver = P_receiver * t_max

    return {
        'E_initial': E_initial,
        'E_initial_ev': E_initial / (photon_energy_ev * e),
        'E_receiver': E_receiver,
        'E_receiver_ev': E_receiver / (photon_energy_ev * e),
        'transmission': T,
        'losses': 1 - T,
        'photon_energy_ev': photon_energy_ev,
        'beam_power': P_initial,
        'beam_intensity': I0
    }


energy_data = calculate_energies()


## Генерация отчета
def generate_report(data):
    report = f"""ОТЧЕТ О МОДЕЛИРОВАНИИ В ВОЗДУШНОЙ СРЕДЕ
Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ПАРАМЕТРЫ СРЕДЫ:
- Показатель преломления: {n_air}
- Проводимость: {sigma_air:.2e} См/м
- Коэффициент пропускания: {data['transmission']:.6f}
- Потери: {data['losses'] * 100:.6f}%

ЭНЕРГЕТИЧЕСКИЕ ПАРАМЕТРЫ:
- Излученная энергия: {data['E_initial_ev']:.4e} эВ
- Энергия на приемнике: {data['E_receiver_ev']:.4e} эВ
- Потери энергии: {(data['E_initial_ev'] - data['E_receiver_ev']):.4e} эВ

ХАРАКТЕРИСТИКИ ИЗЛУЧЕНИЯ:
- Энергия фотона: {data['photon_energy_ev']:.4f} эВ
- Начальная мощность: {data['beam_power']:.4e} Вт
- Начальная интенсивность: {data['beam_intensity']:.4e} Вт/м²
"""
    return report


report = generate_report(energy_data)
print(report)

with open('air_laser_report.txt', 'w') as f:
    f.write(report)

## Визуализация
plt.figure(figsize=(15, 10))

# График интенсивности
plt.subplot(2, 2, 1)
plt.plot(t * 1e9, intensity(0, t))
plt.title('Интенсивность у лазера')
plt.xlabel('Время (нс)')
plt.ylabel('Интенсивность (Вт/м²)')

# График затухания
plt.subplot(2, 2, 2)
z_plot = np.linspace(0, distance, 100)
plt.plot(z_plot, [intensity(z, 0) for z in z_plot])
plt.title('Затухание интенсивности')
plt.xlabel('Расстояние (м)')
plt.ylabel('Интенсивность (Вт/м²)')

# Энергетический баланс
plt.subplot(2, 2, 3)
plt.text(0.1, 0.5,
         f"ЭНЕРГЕТИЧЕСКИЙ БАЛАНС\n\n"
         f"Излученная: {energy_data['E_initial_ev']:.2e} эВ\n"
         f"Принятая: {energy_data['E_receiver_ev']:.2e} эВ\n"
         f"Потери: {energy_data['E_initial_ev'] - energy_data['E_receiver_ev']:.2e} эВ\n"
         f"({energy_data['losses'] * 100:.4f}%)",
         bbox=dict(facecolor='white', alpha=0.8))
plt.axis('off')

plt.tight_layout()
plt.savefig('air_laser_results.png', dpi=300)
plt.show()