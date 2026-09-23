import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, hbar, epsilon_0
from datetime import datetime

# Параметры лазера на GaAs
wavelength = 940e-9  # Длина волны в метрах (типичная для GaAs)
frequency = c / wavelength
omega = 2 * np.pi * frequency
E0 = 10.0  # Амплитуда электрического поля в В/м

# Параметры системы
distance = 1000.0  # Расстояние до приемника в метрах
beam_diameter = 0.002  # Диаметр пучка в метрах (2 мм)
beam_area = np.pi * (beam_diameter / 2) ** 2  # Площадь пучка

# Временные параметры
t_max = 10e-9  # Время моделирования
num_t_points = 1000
t = np.linspace(0, t_max, num_t_points)

# Пространственные параметры
num_z_points = 1000
z = np.linspace(0, distance, num_z_points)


# Моделирование волнового уравнения (решение в виде плоской волны)
def electric_field(z, t):
    return E0 * np.cos(omega * t - (omega / c) * z)


# Расчет интенсивности (I ~ |E|^2)
def intensity(z, t):
    return 0.5 * c * epsilon_0 * electric_field(z, t) ** 2


# Энергия фотона в эВ
photon_energy_ev = (hbar * omega) / e


# Расчет энергий
def calculate_energies():
    # Интенсивность в начальной точке
    I0 = intensity(0, 0)  # Берем максимальное значение (при t=0)

    # Полная излученная мощность
    P_initial = I0 * beam_area

    # Энергия излученная
    E_initial = P_initial * t_max

    # Энергия на приемнике (поскольку потерь в вакууме нет)
    E_receiver = E_initial

    # Энергия в эВ
    E_initial_ev = E_initial / (photon_energy_ev * e)
    E_receiver_ev = E_receiver / (photon_energy_ev * e)

    # Количество фотонов
    num_photons = E_initial / (hbar * omega)

    return {
        'E_initial': E_initial,
        'E_initial_ev': E_initial_ev,
        'E_receiver': E_receiver,
        'E_receiver_ev': E_receiver_ev,
        'num_photons': num_photons,
        'photon_energy_ev': photon_energy_ev,
        'beam_power': P_initial,
        'beam_intensity': I0
    }


# Расчет всех энергетических параметров
energy_data = calculate_energies()


# Функция для создания полного отчета
def generate_energy_report(data):
    report = f"""ЭНЕРГЕТИЧЕСКИЙ ОТЧЕТ ДЛЯ ПОЛУПРОВОДНИКОВОГО ЛАЗЕРА GaAs
Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ПАРАМЕТРЫ СИСТЕМЫ:
- Расстояние до приемника: {distance} м
- Длина волны излучения: {wavelength * 1e9:.2f} нм
- Частота излучения: {frequency / 1e12:.2f} ТГц
- Энергия фотона: {data['photon_energy_ev']:.4f} эВ
- Диаметр пучка: {beam_diameter * 1000:.2f} мм
- Площадь пучка: {beam_area:.2e} м²
- Время моделирования: {t_max * 1e9:.2f} нс

ЭНЕРГЕТИЧЕСКИЕ ПАРАМЕТРЫ:
1. Излученная энергия:
   - В абсолютных единицах: {data['E_initial']:.4e} Дж
   - В электрон-вольтах: {data['E_initial_ev']:.4e} эВ
   - Количество фотонов: {data['num_photons']:.4e}

2. Энергия на приемнике:
   - В абсолютных единицах: {data['E_receiver']:.4e} Дж
   - В электрон-вольтах: {data['E_receiver_ev']:.4e} эВ
   - Потери в вакууме: 0.00 Дж (0.00%)

3. Мощность и интенсивность:
   - Начальная мощность пучка: {data['beam_power']:.4e} Вт
   - Начальная интенсивность: {data['beam_intensity']:.4e} Вт/м²

ЭНЕРГЕТИЧЕСКОЕ УРАВНЕНИЕ БАЛАНСА:
E_излученная = E_дошедшая + E_потерянная + E_отраженная
{data['E_initial_ev']:.4e} эВ = {data['E_receiver_ev']:.4e} эВ + 0.00 эВ + 0.00 эВ

ВЫВОД:
При распространении в вакууме на расстояние {distance} м:
- Энергия полностью сохраняется
- Потери отсутствуют
- Отраженная энергия отсутствует
"""
    return report


# Генерация и вывод отчета
energy_report = generate_energy_report(energy_data)
print(energy_report)

# Сохранение отчета в файл
with open('laser_energy_report.txt', 'w', encoding='utf-8') as f:
    f.write(energy_report)

# Визуализация
plt.figure(figsize=(15, 10))

# 1. График интенсивности излучения в начальной точке (z=0)
plt.subplot(2, 2, 1)
I0 = intensity(0, t)
plt.plot(t * 1e9, I0)
plt.title('Интенсивность излучения у лазера (z=0)')
plt.xlabel('Время (нс)')
plt.ylabel('Интенсивность (Вт/м²)')
plt.grid()

# 2. График интенсивности излучения на расстоянии 1000 м
plt.subplot(2, 2, 2)
I_receiver = intensity(distance, t)
plt.plot(t * 1e9, I_receiver)
plt.title(f'Интенсивность излучения на расстоянии {distance} м')
plt.xlabel('Время (нс)')
plt.ylabel('Интенсивность (Вт/м²)')
plt.grid()

# 3. График потерь энергии в вакууме (должен быть нулевым)
plt.subplot(2, 2, 3)
losses = np.zeros_like(z)
plt.plot(z, losses)
plt.title('Потери энергии в вакууме')
plt.xlabel('Расстояние (м)')
plt.ylabel('Потери энергии (отн. ед.)')
plt.grid()

# 4. Энергетическое уравнение
plt.subplot(2, 2, 4)
plt.text(0.1, 0.3,
         f'ЭНЕРГЕТИЧЕСКИЙ БАЛАНС\n\n'
         f'Излученная энергия:\n{energy_data["E_initial_ev"]:.4e} эВ\n\n'
         f'Энергия на приемнике:\n{energy_data["E_receiver_ev"]:.4e} эВ\n\n'
         f'Потери в вакууме:\n0.00 эВ\n\n'
         f'Отраженная энергия:\n0.00 эВ\n\n'
         f'Баланс:\n{energy_data["E_initial_ev"]:.4e} = '
         f'{energy_data["E_receiver_ev"]:.4e} + 0.00 + 0.00',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.axis('off')

plt.tight_layout()
plt.savefig('laser_energy_analysis.png', dpi=300)
plt.show()