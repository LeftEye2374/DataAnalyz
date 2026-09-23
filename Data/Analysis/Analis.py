import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv('cleaned_hh_ru_dataset.csv')
data_copy = data.copy()

# Группировка данных по профессиям и расчет средней ожидаемой зарплаты
avg_salary_by_profession = data.groupby('profession')['expected_salary'].mean().sort_values(ascending=False)
print(f'Группировка данных по профессиям и расчет средней ожидаемой зарплаты: {avg_salary_by_profession}')
print('_____________________________________________________________________')

# Визуализация
plt.figure(figsize=(12, 8))
avg_salary_by_profession.head(20).plot(kind='bar', color='skyblue')  # Топ-20 профессий
plt.title('Средняя ожидаемая зарплата по профессиям', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Группировка данных по регионам и расчет средней зарплаты
avg_salary_by_region = data.groupby('resume_region')['expected_salary'].mean().sort_values(ascending=False)
print(f'Группировка данных по регионам и расчет средней зарплаты: {avg_salary_by_region}')
print('_____________________________________________________________________')

# Визуализация
plt.figure(figsize=(12, 8))
avg_salary_by_region.head(20).plot(kind='bar', color='lightgreen')  # Топ-20 регионов
plt.title('Средняя ожидаемая зарплата по регионам', fontsize=16)
plt.xlabel('Регион', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Группировка данных по уровню образования и расчет средней зарплаты
avg_salary_by_education = data.groupby('education_level')['expected_salary'].mean().sort_values(ascending=False)
print(f'Группировка данных по уровню образования и расчет средней зарплаты: {avg_salary_by_education}')
print('_____________________________________________________________________')


# Визуализация
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_salary_by_education.index, y=avg_salary_by_education.values, palette='viridis')
plt.title('Средняя ожидаемая зарплата по уровню образования', fontsize=16)
plt.xlabel('Уровень образования', fontsize=14)
plt.ylabel('Средняя зарплата', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Расчет средней вилки зарплат (compensation_from и compensation_to)
data['average_compensation'] = (data['compensation_from'] + data['compensation_to']) / 2

# Группировка по профессиям и расчет средней вилки зарплат
avg_compensation_by_profession = data.groupby('profession')['average_compensation'].mean().sort_values(ascending=False)
print(f'Группировка по профессиям и расчет средней вилки зарплат: {avg_compensation_by_profession}')
print('_____________________________________________________________________')

# Визуализация
plt.figure(figsize=(12, 8))
avg_compensation_by_profession.head(20).plot(kind='bar', color='purple')  # Топ-20 профессий
plt.title('Средняя вилка зарплат по профессиям', fontsize=16)
plt.xlabel('Профессия', fontsize=14)
plt.ylabel('Средняя вилка зарплат', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()