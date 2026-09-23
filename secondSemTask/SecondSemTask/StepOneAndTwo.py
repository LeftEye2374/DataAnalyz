import pandas as pd

# Загрузка данных
df = pd.read_csv('auto-mpg.csv')

# 1. Преобразование столбца 'horsepower' в числовой формат
df['horsepower'] = pd.to_numeric(df['horsepower'].replace('?', pd.NA), errors='coerce')

# 2. Обработка пропусков
median_horsepower = df['horsepower'].median()
df['horsepower'].fillna(median_horsepower, inplace=True)

# 3. Обработка дубликатов
df.drop_duplicates(inplace=True)

# Вывод информации о типах данных
print("\nТипы данных в датасете:")
print(df.dtypes)

# Определение типов шкал измерений
print("\nТипы шкал измерений для каждого признака:")
scale_types = {
    'mpg': 'Относительная (ratio)',
    'cylinders': 'Порядковая (ordinal)',
    'displacement': 'Относительная (ratio)',
    'horsepower': 'Относительная (ratio)',
    'weight': 'Относительная (ratio)',
    'acceleration': 'Относительная (ratio)',
    'model year': 'Интервальная (interval)',
    'origin': 'Номинальная (nominal)',
    'car name': 'Номинальная (nominal)'
}

for column in df.columns:
    print(f"{column}: {scale_types.get(column, 'Не определено')}")

# Дополнительная информация о данных
print("\nДополнительная информация:")
print(f"Всего строк: {len(df)}")
print(f"Количество числовых признаков: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
print(f"Количество категориальных признаков: {len(df.select_dtypes(include=['object']).columns)}")

# Сохранение очищенного датасета
df.to_csv('auto-mpg-cleaned.csv', index=False)
print("\nОчистка данных завершена. Результат сохранен в 'auto-mpg-cleaned.csv'.")