import pandas as pd
import scipy.stats as stats
from scipy.stats import f_oneway, kruskal, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns


class DataRelationshipAnalyzer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Определение шкал измерений для каждого признака
        self.scale_types = {
            'mpg': 'ratio',
            'cylinders': 'ordinal',
            'displacement': 'ratio',
            'horsepower': 'ratio',
            'weight': 'ratio',
            'acceleration': 'ratio',
            'model year': 'interval',
            'origin': 'nominal',
            'car name': 'nominal'
        }

    def analyze_relationship(self, col1, col2):
        """Анализирует взаимосвязь между двумя переменными"""
        scale1 = self.scale_types.get(col1, 'unknown')
        scale2 = self.scale_types.get(col2, 'unknown')

        print(f"\nАнализ взаимосвязи между '{col1}' ({scale1}) и '{col2}' ({scale2}):")

        # Случай 1: Обе переменные количественные или порядковые
        if (scale1 in ['ratio', 'interval', 'ordinal'] and
                scale2 in ['ratio', 'interval', 'ordinal']):
            self._analyze_numeric_numeric(col1, col2)

        # Случай 2: Одна количественная/порядковая, другая категориальная
        elif (scale1 in ['ratio', 'interval', 'ordinal'] and scale2 == 'nominal'):
            self._analyze_numeric_categorical(col1, col2)
        elif (scale1 == 'nominal' and scale2 in ['ratio', 'interval', 'ordinal']):
            self._analyze_numeric_categorical(col2, col1)

        # Случай 3: Обе категориальные
        elif scale1 == 'nominal' and scale2 == 'nominal':
            self._analyze_categorical_categorical(col1, col2)

        else:
            print("Неизвестная комбинация типов переменных")

    def _analyze_numeric_numeric(self, col1, col2):
        """Анализ связи между двумя числовыми/порядковыми переменными"""
        # Визуализация
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x=col1, y=col2)
        plt.title(f"Scatter plot: {col1} vs {col2}")
        plt.show()

        # Корреляционный анализ
        pearson_corr, pearson_p = stats.pearsonr(self.df[col1], self.df[col2])
        spearman_corr, spearman_p = stats.spearmanr(self.df[col1], self.df[col2])

        print("\nКорреляционный анализ:")
        print(f"Корреляция Пирсона: {pearson_corr:.3f} (p-value: {pearson_p:.3f})")
        print(f"Корреляция Спирмена: {spearman_corr:.3f} (p-value: {spearman_p:.3f})")

        # Интерпретация
        self._interpret_correlation(pearson_corr, spearman_corr)

    def _analyze_numeric_categorical(self, num_col, cat_col):
        """Анализ связи между числовой и категориальной переменной"""
        # Визуализация
        plt.figure(figsize=(10, 6))
        if len(self.df[cat_col].unique()) <= 5:
            sns.boxplot(data=self.df, x=cat_col, y=num_col)
        else:
            sns.violinplot(data=self.df, x=cat_col, y=num_col)
        plt.title(f"Распределение {num_col} по категориям {cat_col}")
        plt.xticks(rotation=45)
        plt.show()

        # Статистический анализ
        groups = [group[1][num_col].values for group in self.df.groupby(cat_col)]

        if len(groups) == 2:
            # t-тест для двух групп
            stat, p = stats.ttest_ind(*groups, equal_var=False)
            test_name = "t-тест (Уэлча)"
        else:
            # ANOVA или Краскела-Уоллиса для нескольких групп
            if stats.normaltest(self.df[num_col]).pvalue > 0.05:
                stat, p = f_oneway(*groups)
                test_name = "ANOVA"
            else:
                stat, p = kruskal(*groups)
                test_name = "Краскела-Уоллиса"

        print(f"\nРезультаты теста {test_name}:")
        print(f"Статистика: {stat:.3f}, p-value: {p:.3f}")

        # Интерпретация
        if p < 0.05:
            print("Есть статистически значимые различия между группами (p < 0.05)")
        else:
            print("Нет статистически значимых различий между группами (p >= 0.05)")

    def _analyze_categorical_categorical(self, col1, col2):
        """Анализ связи между двумя категориальными переменными"""
        # Визуализация - таблица сопряженности
        contingency_table = pd.crosstab(self.df[col1], self.df[col2])

        print("\nТаблица сопряженности:")
        print(contingency_table)

        # Визуализация - тепловая карта
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Связь между {col1} и {col2}")
        plt.show()

        # Хи-квадрат тест
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        print("\nРезультаты теста хи-квадрат:")
        print(f"Хи-квадрат: {chi2:.3f}, p-value: {p:.3f}, степени свободы: {dof}")

        # Интерпретация
        if p < 0.05:
            print("Есть статистически значимая связь между переменными (p < 0.05)")
        else:
            print("Нет статистически значимой связи между переменными (p >= 0.05)")

    def _interpret_correlation(self, pearson, spearman):
        """Интерпретация корреляции"""
        print("\nИнтерпретация корреляции:")

        corr = pearson  # Используем Пирсона для интерпретации

        abs_corr = abs(corr)
        if abs_corr >= 0.9:
            strength = "очень сильная"
        elif abs_corr >= 0.7:
            strength = "сильная"
        elif abs_corr >= 0.5:
            strength = "умеренная"
        elif abs_corr >= 0.3:
            strength = "слабая"
        else:
            strength = "очень слабая или отсутствует"

        direction = "положительная" if corr > 0 else "отрицательная"

        print(f"{strength} {direction} корреляция")

    def comprehensive_analysis(self):
        """Комплексный анализ всех возможных пар переменных"""
        print("\n" + "=" * 50)
        print("КОМПЛЕКСНЫЙ АНАЛИЗ ВЗАИМОСВЯЗЕЙ")
        print("=" * 50 + "\n")

        # Анализ числовых переменных между собой
        print("\n" + "-" * 20)
        print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ЧИСЛОВЫХ ПЕРЕМЕННЫХ")
        print("-" * 20 + "\n")

        numeric_pairs = [(col1, col2) for i, col1 in enumerate(self.numeric_cols)
                         for j, col2 in enumerate(self.numeric_cols) if i < j]

        for col1, col2 in numeric_pairs:
            self.analyze_relationship(col1, col2)

        # Анализ числовых и категориальных переменных
        print("\n" + "-" * 20)
        print("АНАЛИЗ ЧИСЛОВЫХ И КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ")
        print("-" * 20 + "\n")

        for num_col in self.numeric_cols:
            for cat_col in self.categorical_cols:
                if len(self.df[cat_col].unique()) <= 10:  # Ограничим число категорий
                    self.analyze_relationship(num_col, cat_col)

        # Анализ категориальных переменных между собой
        print("\n" + "-" * 20)
        print("АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ")
        print("-" * 20 + "\n")

        categorical_pairs = [(col1, col2) for i, col1 in enumerate(self.categorical_cols)
                             for j, col2 in enumerate(self.categorical_cols) if i < j]

        for col1, col2 in categorical_pairs:
            if len(self.df[col1].unique()) <= 10 and len(self.df[col2].unique()) <= 10:
                self.analyze_relationship(col1, col2)

        print("\nАнализ завершен!")

# Пример использования:
df = pd.read_csv('auto-mpg-cleaned.csv')
analyzer = DataRelationshipAnalyzer(df)

# Для анализа конкретной пары переменных:
analyzer.analyze_relationship('mpg', 'origin')

#Для комплексного анализа всех переменных:
analyzer.comprehensive_analysis()