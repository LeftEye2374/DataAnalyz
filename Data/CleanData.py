import pandas as pd
import numpy as np

# Загрузка датасета
df = pd.read_csv('hh_ru_dataset.csv')

# Удаление дубликатов
df.drop_duplicates(inplace=True)

# Обработка пропущенных значений для числовых столбцов
numeric_columns = ['year_of_birth', 'expected_salary', 'work_experience_months', 'compensation_from', 'compensation_to']
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        # Используем более явное присваивание
        df[col] = df[col].fillna(df[col].median())

# Обработка пропущенных значений для категориальных столбцов
categorical_columns = [
    'topic_id', 'topic_creation_date', 'initial_state', 'final_state', 'resume_id', 'resume_creation_date',
    'profession', 'gender', 'resume_region', 'education_level', 'relocation_status', 'business_trip_readiness',
    'work_schedule', 'resume_employment_type', 'resume_skills_list', 'vacancy_id', 'vacancy_creation_date',
    'vacancy_region', 'work_schedule.1', 'vacancy_employment_type', 'vacancy_skills_list'
]
for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        # Используем более явное присваивание
        df[col] = df[col].fillna('Unknown')

# Преобразование типов данных
df['year_of_birth'] = df['year_of_birth'].astype(int)
df['expected_salary'] = df['expected_salary'].astype(float)
df['work_experience_months'] = df['work_experience_months'].astype(int)
df['compensation_from'] = df['compensation_from'].astype(float)
df['compensation_to'] = df['compensation_to'].astype(float)

# Удаление аномалий (возраст меньше 18 или больше 100 лет)
current_year = pd.Timestamp.now().year
df = df[(df['year_of_birth'] >= 1923) & (df['year_of_birth'] <= 2005)]

# Создание нового признака "age"
df['age'] = current_year - df['year_of_birth']

# Проверка результатов очистки
print("Информация о датасете после очистки:")
print(df.info())
print("\nКоличество пропущенных значений после очистки:")
print(df.isnull().sum())

# Сохранение очищенного датасета (при необходимости)
df.to_csv('cleaned_hh_ru_dataset.csv', index=False)