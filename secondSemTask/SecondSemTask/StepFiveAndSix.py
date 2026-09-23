import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway, pearsonr, spearmanr, shapiro, levene, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Загрузка данных
df = pd.read_csv('auto-mpg-cleaned.csv')


# =============================================
# Шаг 5: Проверка гипотез
# =============================================

class HypothesisTester:
    def __init__(self, df):
        self.df = df

    def test_hypothesis_1(self):
        """Гипотеза о связи мощности двигателя и расхода топлива"""
        print("\n" + "=" * 50)
        print("Шаг 5: Проверка гипотез")
        print("=" * 50 + "\n")
        print("Гипотеза 1: Связь между мощностью и расходом топлива")

        # 1. Формулировка гипотезы
        print("\nНулевая гипотеза H0: Нет корреляции между horsepower и mpg")
        print("Альтернативная гипотеза H1: Существует отрицательная корреляция\n")

        # 2. Визуализация
        plt.figure(figsize=(10, 6))
        sns.regplot(data=self.df, x='horsepower', y='mpg',
                    scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        plt.title("Зависимость расхода топлива от мощности двигателя")
        plt.xlabel("Мощность (horsepower)")
        plt.ylabel("Расход топлива (mpg)")
        plt.show()

        # 3. Выбор критерия
        print("Критерии:")
        print("- Корреляция Пирсона для линейной зависимости")
        print("- Корреляция Спирмена для монотонной зависимости\n")

        # 4. Применение критериев
        pearson_corr, pearson_p = pearsonr(self.df['horsepower'], self.df['mpg'])
        spearman_corr, spearman_p = spearmanr(self.df['horsepower'], self.df['mpg'])

        print("Результаты:")
        print(f"Корреляция Пирсона: r={pearson_corr:.3f}, p={pearson_p:.3e}")
        print(f"Корреляция Спирмена: ρ={spearman_corr:.3f}, p={spearman_p:.3e}")

        # 5. Вывод
        print("\nВывод:")
        if pearson_p < 0.05 and spearman_p < 0.05:
            print("Отвергаем H0: существует статистически значимая отрицательная корреляция")
            print("Сила связи: умеренная (коэффициенты около -0.8)")
        else:
            print("Нет оснований отвергать H0")
        print("\n" + "-" * 50)

    def test_hypothesis_2(self):
        """Гипотеза о различии расхода топлива по регионам"""
        print("\nГипотеза 2: Различия в расходе топлива по регионам")

        # 1. Формулировка гипотезы
        print("\nНулевая гипотеза H0: Средний расход (mpg) одинаков для всех регионов")
        print("Альтернативная гипотеза H1: Средний расход различается\n")

        # 2. Визуализация
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='origin', y='mpg')
        plt.title("Распределение расхода по регионам производства")
        plt.xlabel("Регион (1-США, 2-Европа, 3-Азия)")
        plt.ylabel("Расход топлива (mpg)")
        plt.show()

        # 3. Проверка условий
        # Проверка нормальности
        _, p_shapiro = shapiro(self.df['mpg'])
        print(f"Тест Шапиро-Уилка на нормальность: p={p_shapiro:.3f}")

        # Проверка гомогенности дисперсий
        groups = [group[1]['mpg'].values for group in self.df.groupby('origin')]
        _, p_levene = levene(*groups)
        print(f"Тест Левена на гомогенность: p={p_levene:.3f}")

        # 4. Выбор критерия
        if p_shapiro > 0.05 and p_levene > 0.05:
            print("\nИспользуем ANOVA (условия выполнены)")
            test_name = "ANOVA"
            f_stat, p_value = f_oneway(*groups)
            print(f"F={f_stat:.3f}, p={p_value:.3e}")
        else:
            print("\nИспользуем тест Краскела-Уоллиса (условия не выполнены)")
            test_name = "Краскела-Уоллиса"
            h_stat, p_value = kruskal(*groups)
            print(f"H={h_stat:.3f}, p={p_value:.3e}")

        # 5. Вывод
        print("\nВывод:")
        if p_value < 0.05:
            print("Отвергаем H0: существуют статистически значимые различия")

            # Post-hoc анализ
            print("\nPost-hoc тест Тьюки:")
            tukey = mc.pairwise_tukeyhsd(self.df['mpg'], self.df['origin'])
            print(tukey)
        else:
            print("Нет оснований отвергать H0")
        print("\n" + "=" * 50)


# =============================================
# Шаг 6: Построение регрессионных моделей
# =============================================

class RegressionModeler:
    def __init__(self, df, target='mpg'):
        self.df = df
        self.target = target
        self._prepare_data()

    def _prepare_data(self):
        """Подготовка данных"""
        # Выбор числовых признаков
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.features = [col for col in numeric_cols if col != self.target]

        # Разделение на train/test
        X = self.df[self.features]
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    def build_polynomial_model(self, features=['horsepower', 'weight'], degree=2):
        """Построение полиномиальной модели"""
        print(f"\n=== Полиномиальная регрессия (степень {degree}) ===")

        # Создание полиномиальных признаков
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(self.X_train[features])
        X_test_poly = poly.transform(self.X_test[features])

        # Получение имен признаков
        feature_names = poly.get_feature_names_out(features)

        # Создание DataFrame с правильными индексами
        X_train_poly = pd.DataFrame(X_train_poly, columns=feature_names, index=self.X_train.index)
        X_test_poly = pd.DataFrame(X_test_poly, columns=feature_names, index=self.X_test.index)

        # Добавление константы
        X_train_poly = sm.add_constant(X_train_poly)
        X_test_poly = sm.add_constant(X_test_poly)

        # Обучение модели
        model = sm.OLS(self.y_train, X_train_poly).fit()

        # Прогнозирование
        y_pred = model.predict(X_test_poly)

        # Оценка
        self._evaluate_model(model, y_pred)
        self._plot_results(y_pred, f"Полиномиальная (степень {degree})")

        return model

    def build_polynomial_model(self, features=['horsepower', 'weight'], degree=2):
        """Построение полиномиальной модели"""
        print(f"\n=== Полиномиальная регрессия (степень {degree}) ===")

        # Создание полиномиальных признаков
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(self.X_train[features])
        X_test_poly = poly.transform(self.X_test[features])

        # Получение имен признаков
        feature_names = poly.get_feature_names_out(features)
        X_train_poly = pd.DataFrame(X_train_poly, columns=feature_names)
        X_test_poly = pd.DataFrame(X_test_poly, columns=feature_names)

        # Добавление константы
        X_train_poly = sm.add_constant(X_train_poly)
        X_test_poly = sm.add_constant(X_test_poly)

        # Обучение модели
        model = sm.OLS(self.y_train, X_train_poly).fit()

        # Прогнозирование
        y_pred = model.predict(X_test_poly)

        # Оценка
        self._evaluate_model(model, y_pred)
        self._plot_results(y_pred, f"Полиномиальная (степень {degree})")

        return model

    def _evaluate_model(self, model, y_pred):
        """Оценка качества модели"""
        # Метрики
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Вывод результатов
        print(model.summary())
        print("\nМетрики качества:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")

    def _plot_results(self, y_pred, title):
        """Визуализация результатов"""
        residuals = self.y_test - y_pred

        plt.figure(figsize=(12, 5))
        plt.suptitle(title)

        # График остатков
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Предсказанные значения")
        plt.ylabel("Остатки")

        # QQ-plot
        plt.subplot(1, 2, 2)
        sm.qqplot(residuals, line='s', ax=plt.gca())
        plt.tight_layout()
        plt.show()

    def compare_models(self, model1, model2, model1_name="Модель 1", model2_name="Модель 2"):
        """Сравнение двух моделей"""
        print("\n=== Сравнение моделей ===")

        # Прогнозы
        y_pred1 = model1.predict(sm.add_constant(self.X_test))
        y_pred2 = model2.predict(sm.add_constant(self.X_test))

        # Метрики
        metrics = {
            'Metric': ['R²', 'RMSE', 'MAE'],
            model1_name: [
                r2_score(self.y_test, y_pred1),
                np.sqrt(mean_squared_error(self.y_test, y_pred1)),
                mean_absolute_error(self.y_test, y_pred1)
            ],
            model2_name: [
                r2_score(self.y_test, y_pred2),
                np.sqrt(mean_squared_error(self.y_test, y_pred2)),
                mean_absolute_error(self.y_test, y_pred2)
            ]
        }

        print(pd.DataFrame(metrics).to_string(index=False))

        # Определение лучшей модели
        best_model = model1_name if metrics[model1_name][0] > metrics[model2_name][0] else model2_name
        print(f"\nЛучшая модель: {best_model} (по R²)")


# =============================================
# Основной блок выполнения
# =============================================

if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_csv('auto-mpg-cleaned.csv')

    # Шаг 5: Проверка гипотез
    tester = HypothesisTester(df)
    tester.test_hypothesis_1()  # Связь мощности и расхода
    tester.test_hypothesis_2()  # Различия по регионам

    # Шаг 6: Построение моделей
    modeler = RegressionModeler(df)

    # Линейная модель
    linear_model = modeler.build_linear_model(features=['horsepower', 'weight', 'acceleration'])

    # Полиномиальная модель
    poly_model = modeler.build_polynomial_model(features=['horsepower', 'weight'], degree=2)

    # Сравнение моделей
    modeler.compare_models(linear_model, poly_model,
                           model1_name="Линейная",
                           model2_name="Полиномиальная")