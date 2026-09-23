import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns
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

    def _determine_plot_type(self, col):
        """Определяет тип графика на основе шкалы измерения"""
        scale_type = self.scale_types.get(col, 'unknown')

        if scale_type == 'nominal':
            if len(self.df[col].unique()) <= 10:
                return 'countplot'
            else:
                return 'barplot_top20'
        elif scale_type == 'ordinal':
            return 'histplot'
        elif scale_type in ['interval', 'ratio']:
            if len(self.df[col].unique()) <= 10:
                return 'boxplot'
            else:
                return 'histplot'
        else:
            return 'histplot'  # fallback

    def visualize_column(self, col, figsize=(10, 6)):
        """Визуализирует один столбец"""
        plot_type = self._determine_plot_type(col)

        plt.figure(figsize=figsize)
        plt.title(f'Визуализация для "{col}" ({self.scale_types.get(col, "unknown")} scale)')

        if plot_type == 'countplot':
            sns.countplot(data=self.df, x=col)
            plt.xticks(rotation=45)
        elif plot_type == 'barplot_top20':
            top20 = self.df[col].value_counts().nlargest(20)
            sns.barplot(x=top20.values, y=top20.index)
            plt.xlabel('Count')
        elif plot_type == 'histplot':
            sns.histplot(data=self.df, x=col, kde=True)
        elif plot_type == 'boxplot':
            sns.boxplot(data=self.df, x=col)

        plt.tight_layout()
        plt.show()

    def visualize_numeric_relationships(self, target_col=None):
        """Визуализирует взаимосвязи между числовыми переменными"""
        if target_col:
            # Pairplot с выделением целевой переменной
            sns.pairplot(self.df, vars=self.numeric_cols, hue=target_col)
            plt.suptitle(f'Pairplot с выделением по "{target_col}"', y=1.02)
        else:
            # Тепловая карта корреляций
            plt.figure(figsize=(12, 8))
            corr = self.df[self.numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
            plt.title('Матрица корреляций числовых признаков')

        plt.tight_layout()
        plt.show()

    def visualize_categorical_relationships(self, target_col=None):
        """Визуализирует взаимосвязи категориальных переменных"""
        if not target_col:
            print("Необходимо указать целевую переменную для анализа категориальных данных")
            return

        for col in self.categorical_cols:
            if col != target_col:
                plt.figure(figsize=(10, 6))
                if len(self.df[col].unique()) > 10:
                    # Для переменных с большим числом категорий берем топ-10
                    top_categories = self.df[col].value_counts().nlargest(10).index
                    temp_df = self.df[self.df[col].isin(top_categories)]
                else:
                    temp_df = self.df

                sns.countplot(data=temp_df, x=col, hue=target_col)
                plt.title(f'Распределение "{col}" по "{target_col}"')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

    def comprehensive_visualization(self, target_col=None):
        """Комплексная визуализация всех переменных"""

        # 1. Визуализация отдельных переменных

        for col in self.df.columns:
            self.visualize_column(col)

        # 2. Анализ числовых взаимосвязей
        self.visualize_numeric_relationships(target_col)

        # 3. Анализ категориальных взаимосвязей
        if target_col and target_col in self.categorical_cols:
            self.visualize_categorical_relationships(target_col)


# Пример использования:
# Загрузка данных
df = pd.read_csv('auto-mpg-cleaned.csv')

# Создание визуализатора
visualizer = DataVisualizer(df)

# Комплексная визуализация
visualizer.comprehensive_visualization(target_col='origin')

# Или выборочная визуализация:
# visualizer.visualize_column('horsepower')
# visualizer.visualize_numeric_relationships()