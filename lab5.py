"""
Этот скрипт выполняет полный цикл проекта машинного обучения для задачи бинарной классификации – предсказания дефолта по кредиту. Включает в себя:
1. Загрузка и предобработку данных
2. Исследовательский анализ данных (EDA) с генерацией графиков
3. Обработка пропущенных значений
4. Кодирование категориальных переменных
5. Разделение данных на обучающие и тестовые наборы
6. Создание и обучение модели дерева решений
7. Визуализация дерева решений
8. Оценка модели с помощью различных метрик
9. Визуализация ROC-кривой и Precision-Recall кривой
10. Настройка гиперпараметров с помощью Grid Search и Randomized Search
11. Визуализация результатов настройки гиперпараметров
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Настройка отображения графиков с использованием seaborn
sns.set(style='whitegrid')  # Устанавливаем стиль seaborn

# Настройка параметров matplotlib
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 300

# Игнорирование предупреждений FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_df_memory_usage(df, top_columns=5):
    '''
    Функция для быстрого анализа использования памяти pandas DataFrame.
    Она выводит топ-столбцов по использованию памяти и общее использование.
    '''
    print('Использование памяти ----')
    memory_per_column = df.memory_usage(deep=True) / 1024 ** 2
    print(f'Top {top_columns} столбцов по использованию памяти (MB):')
    print(memory_per_column.sort_values(ascending=False).head(top_columns))
    print(f'Общее использование: {memory_per_column.sum():.4f} MB\n')


def main():
    # -------------------------------
    # 2. Загрузка и предварительный осмотр данных
    # -------------------------------

    # Загрузка данных из CSV файла
    data_path = './dataset/credit_card_default.csv'

    if not os.path.exists(data_path):
        print(f"Файл данных не найден по пути: {data_path}")
        return

    try:
        df = pd.read_csv(data_path, index_col=0, na_values='')
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return

    # Вывод имен столбцов
    print("Имена столбцов DataFrame:")
    print(df.columns.tolist())

    # Если столбцы неправильно распознаны, попытайтесь указать правильный заголовок
    if 'X1' in df.columns:
        print("\nПохоже, что заголовки столбцов не были правильно распознаны. Пытаемся исправить...")
        try:
            # Считаем, что реальный заголовок находится на второй строке (индекс 1)
            df = pd.read_csv(data_path, header=1, index_col=0, na_values='')
            print("Заголовки успешно обновлены.")
            print("Новые имена столбцов:")
            print(df.columns.tolist())
        except Exception as e:
            print(f"Не удалось обновить заголовки столбцов: {e}")
            return

    # Приведение всех имен столбцов к нижнему регистру и замену пробелов на подчёркивания
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # Вывод информации о размере DataFrame
    print(f'DataFrame имеет {len(df)} строк и {df.shape[1]} столбцов.')

    # Вывод первых пяти строк DataFrame
    print("\nПервые пять строк DataFrame:")
    print(df.head())

    # -------------------------------
    # 3. Оптимизация использования памяти
    # -------------------------------

    # Проверка исходного использования памяти
    print("Исходное использование памяти:")
    get_df_memory_usage(df)

    # Явное указание категориальных столбцов
    categorical_features = ['sex', 'education', 'marriage']

    # Преобразование категориальных столбцов в тип 'category'
    df[categorical_features] = df[categorical_features].astype('category')

    # Проверка использования памяти после преобразования
    print("Использование памяти после преобразования категориальных столбцов:")
    get_df_memory_usage(df)

    # -------------------------------
    # 4. Исследовательский анализ данных (EDA)
    # -------------------------------

    # Сводная статистика для числовых переменных
    print("Сводная статистика для числовых переменных:")
    print(df.describe())

    # Сводная статистика для категориальных переменных
    print("\nСводная статистика для категориальных переменных:")
    print(df.describe(include=['category']))

    # 4.1 Распределение возраста по полу
    age_column = 'age'
    sex_column = 'sex'

    if age_column in df.columns and sex_column in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=age_column, hue=sex_column, kde=True, palette='Set1', multiple='stack')
        plt.title('Распределение возраста по полу')
        plt.xlabel('Возраст')
        plt.ylabel('Количество')
        plt.tight_layout()
        plt.savefig('eda_age_distribution_by_sex.png')
        plt.close()
    else:
        print(f"Не удалось найти столбцы для 'age' и/или 'sex'. Найдены столбцы: {age_column}, {sex_column}")

    # 4.2 Парный график выбранных переменных
    selected_columns = ['limit_bal', 'age', 'bill_amt1', 'pay_amt1']
    existing_selected_columns = [col for col in selected_columns if col in df.columns]

    target_column = 'default_payment_next_month'

    if target_column in df.columns:
        sns.pairplot(df[existing_selected_columns + [target_column]], hue=target_column, palette='Set2')
        plt.suptitle('Парный график выбранных переменных', y=1.02)
        plt.tight_layout()
        plt.savefig('eda_pairplot_selected_features.png')
        plt.close()
    else:
        print("Не удалось найти целевой столбец для парного графика.")

    # 4.3 Тепловая карта корреляционной матрицы
    corr_matrix = df.select_dtypes(include=[np.number]).corr()

    # Создание маски для верхнего треугольника
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Корреляционная матрица числовых признаков')
    plt.tight_layout()
    plt.savefig('eda_correlation_heatmap.png')
    plt.close()

    # 4.4 Скрипичный график предельного баланса по уровню образования и полу
    education_column = 'education'
    limit_bal_column = 'limit_bal'

    if education_column in df.columns and sex_column in df.columns and limit_bal_column in df.columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=education_column, y=limit_bal_column, hue=sex_column, data=df, split=True, palette='Set2')
        plt.title('Распределение предельного баланса по уровню образования и полу')
        plt.xlabel('Уровень образования')
        plt.ylabel('Предельный баланс (NT долл.)')
        plt.tight_layout()
        plt.savefig('eda_violinplot_limit_bal_by_education_sex.png')
        plt.close()
    else:
        print(f"Не удалось найти необходимые столбцы для скрипичного графика. Найдены столбцы: {education_column}, {sex_column}, {limit_bal_column}")

    # 4.5 Распределение дефолтов по полу
    if sex_column in df.columns and target_column in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=sex_column, hue=target_column, data=df, palette='Set1')
        plt.title('Распределение дефолтов по полу')
        plt.xlabel('Пол')
        plt.ylabel('Количество')
        plt.legend(title='Дефолт', labels=['Нет', 'Да'])
        plt.tight_layout()
        plt.savefig('eda_default_by_sex.png')
        plt.close()
    else:
        print(f"Не удалось найти столбцы для распределения дефолтов по полу. Найдены столбцы: {sex_column}, {target_column}")

    # 4.6 Распределение дефолтов по уровню образования
    if education_column in df.columns and target_column in df.columns:
        default_by_education = df.groupby(education_column)[target_column].mean().reset_index()

        plt.figure(figsize=(8, 6))
        sns.barplot(x=education_column, y=default_by_education[target_column], data=default_by_education, palette='Set2')
        plt.title('Процент дефолтов по уровню образования')
        plt.xlabel('Уровень образования')
        plt.ylabel('Процент дефолтов')
        plt.tight_layout()
        plt.savefig('eda_default_by_education.png')
        plt.close()
    else:
        print(f"Не удалось найти столбцы для распределения дефолтов по образованию. Найдены столбцы: {education_column}, {target_column}")

    # 4.7 Боксплот предельного баланса по уровню образования и полу
    if education_column in df.columns and sex_column in df.columns and limit_bal_column in df.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=education_column, y=limit_bal_column, hue=sex_column, data=df, palette='Set3')
        plt.title('Распределение предельного баланса по уровню образования и полу')
        plt.xlabel('Уровень образования')
        plt.ylabel('Предельный баланс (NT долл.)')
        plt.legend(title='Пол')
        plt.tight_layout()
        plt.savefig('eda_boxplot_limit_bal_by_education_sex.png')
        plt.close()
    else:
        print(f"Не удалось найти необходимые столбцы для боксплота. Найдены столбцы: {education_column}, {sex_column}, {limit_bal_column}")

    # -------------------------------
    # 5. Обработка пропущенных значений
    # -------------------------------

    # Проверка наличия пропущенных значений
    print("Пропущенные значения по столбцам:")
    print(df.isnull().sum())

    # Отделение признаков и целевой переменной
    X = df.drop(target_column, axis=1) if target_column in df.columns else df.drop('y', axis=1, errors='ignore')
    y = df[target_column] if target_column in df.columns else df['y'] if 'y' in df.columns else None

    if y is None:
        print("Целевая переменная не найдена в DataFrame.")
        return

    # Разделение данных на обучающие и тестовые наборы (стратифицированное)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Определение числовых и категориальных признаков
    categorical_features = ['sex', 'education', 'marriage']
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # Создание конвейера для числовых признаков
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Создание конвейера для категориальных признаков
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # Объединение преобразователей в ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Проверка пропущенных значений после разделения
    print("\nПропущенные значения в обучающем наборе после разделения:")
    print(X_train.isnull().sum())
    print("\nПропущенные значения в тестовом наборе после разделения:")
    print(X_test.isnull().sum())

    # -------------------------------
    # 6. Кодирование категориальных переменных
    # -------------------------------

    # Этот шаг уже выполнен внутри конвейера preprocessor

    # -------------------------------
    # 7. Разделение данных на обучающие и тестовые наборы
    # -------------------------------

    # Этот шаг уже выполнен ранее

    # -------------------------------
    # 8. Создание и обучение модели дерева решений
    # -------------------------------

    # Создание конвейера с классификатором дерева решений
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    # Обучение модели
    print("\nОбучение модели дерева решений...")
    clf.fit(X_train, y_train)
    print("Модель обучена.")

    # Прогнозирование на тестовом наборе
    y_pred = clf.predict(X_test)

    # -------------------------------
    # 9. Визуализация дерева решений
    # -------------------------------

    # Визуализация дерева решений
    plt.figure(figsize=(20, 10))
    tree_model = clf.named_steps['classifier']

    # Получение имен категориальных признаков после one-hot кодирования
    # В sklearn >= 1.0 используется get_feature_names_out()
    try:
        feature_names_cat = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
            'onehot'].get_feature_names_out(categorical_features)
    except AttributeError:
        feature_names_cat = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
            'onehot'].get_feature_names(categorical_features)

    feature_names = list(numeric_features) + list(feature_names_cat)

    plot_tree(tree_model,
              feature_names=feature_names,
              class_names=['Нет дефолта', 'Дефолт'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('Дерево решений')
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png')
    plt.close()

    # -------------------------------
    # 10. Оценка модели
    # -------------------------------

    # Сводный отчет по классификации
    print("\nСводный отчет по классификации:")
    print(classification_report(y_test, y_pred))

    # Получение точности модели
    accuracy = clf.score(X_test, y_test)
    print(f'Точность модели: {accuracy:.4f}')

    # -------------------------------
    # 11. Визуализация ROC-кривой и Precision-Recall кривой
    # -------------------------------

    # Расчет вероятностей предсказания
    y_probs = clf.predict_proba(X_test)[:, 1]

    # 11.1 ROC-кривая
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ложно-положительная ставка')
    plt.ylabel('Истинно-положительная ставка')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()

    # 11.2 Precision-Recall кривая
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR-кривая (AUC = {pr_auc:.2f})')
    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title('Precision-Recall Кривая')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    plt.close()

    # -------------------------------
    # 12. Настройка гиперпараметров с помощью Grid Search и Randomized Search
    # -------------------------------

    # 12.1 Определение сетки параметров для Grid Search
    param_grid = {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [None, 5, 10, 15, 20],
        'classifier__min_samples_leaf': [1, 2, 4, 6, 8],
        # Следующие параметры уже зафиксированы и не меняются, можно убрать их из сетки
        # 'preprocessor__num__imputer__strategy': ['median'],
        # 'preprocessor__cat__imputer__strategy': ['most_frequent'],
        # 'preprocessor__cat__onehot__drop': ['first']
    }

    # Создание GridSearchCV
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=1)

    # Обучение Grid Search
    print("\nНачало Grid Search...")
    grid_search.fit(X_train, y_train)
    print("Grid Search завершен.")

    # Лучшая модель и параметры из Grid Search
    print("\nЛучшая модель из Grid Search:")
    print(grid_search.best_estimator_)
    print("\nЛучшие параметры из Grid Search:")
    print(grid_search.best_params_)

    # 12.2 Визуализация результатов Grid Search
    # Извлечение результатов Grid Search
    results_grid = pd.DataFrame(grid_search.cv_results_)

    # Визуализация точности (recall) в зависимости от max_depth
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='param_classifier__max_depth', y='mean_test_score', data=results_grid, marker='o')
    plt.title('Зависимость Recall от max_depth (Grid Search)')
    plt.xlabel('max_depth')
    plt.ylabel('Mean Recall')
    plt.tight_layout()
    plt.savefig('grid_search_max_depth.png')
    plt.close()

    # 12.3 Создание и запуск Randomized Search

    from scipy.stats import randint

    # Определение распределений для Randomized Search
    param_dist = {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [None] + list(range(5, 21, 5)),
        'classifier__min_samples_leaf': randint(1, 10),
    }

    # Создание RandomizedSearchCV
    random_search = RandomizedSearchCV(clf, param_dist, n_iter=100, cv=5, scoring='recall',
                                       n_jobs=-1, random_state=42, verbose=1)

    # Обучение Randomized Search
    print("\nНачало Randomized Search...")
    random_search.fit(X_train, y_train)
    print("Randomized Search завершен.")

    # Лучшая модель и параметры из Randomized Search
    print("\nЛучшая модель из Randomized Search:")
    print(random_search.best_estimator_)
    print("\nЛучшие параметры из Randomized Search:")
    print(random_search.best_params_)

    # 12.4 Визуализация результатов Randomized Search
    # Извлечение результатов Randomized Search
    results_random = pd.DataFrame(random_search.cv_results_)

    # Визуализация распределения Recall
    plt.figure(figsize=(10, 6))
    sns.histplot(results_random['mean_test_score'], bins=20, kde=True, color='green')
    plt.title('Распределение Recall в Randomized Search')
    plt.xlabel('Mean Recall')
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.savefig('random_search_recall_distribution.png')
    plt.close()

    # -------------------------------
    # 13. Визуализация результатов настройки гиперпараметров
    # -------------------------------

    # 13.1 Сравнение Grid Search и Randomized Search
    # Получение наилучших результатов
    best_grid = grid_search.best_score_
    best_random = random_search.best_score_

    # Сравнение точности
    methods = ['Grid Search', 'Randomized Search']
    scores = [best_grid, best_random]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=methods, y=scores, palette='viridis')
    plt.title('Сравнение Recall между Grid Search и Randomized Search')
    plt.ylabel('Mean Recall')
    plt.ylim(0, 1)
    for index, value in enumerate(scores):
        plt.text(index, value + 0.01, f"{value:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('grid_vs_random_search_comparison.png')
    plt.close()

    # 13.2 Визуализация ROC-кривых для лучших моделей
    # ROC для лучшей модели из Grid Search
    y_probs_grid = grid_search.predict_proba(X_test)[:, 1]
    fpr_grid, tpr_grid, _ = roc_curve(y_test, y_probs_grid)
    roc_auc_grid = auc(fpr_grid, tpr_grid)

    # ROC для лучшей модели из Randomized Search
    y_probs_random = random_search.predict_proba(X_test)[:, 1]
    fpr_random, tpr_random, _ = roc_curve(y_test, y_probs_random)
    roc_auc_random = auc(fpr_random, tpr_random)

    # Визуализация ROC-кривых
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_grid, tpr_grid, color='darkorange', lw=2, label=f'Grid Search ROC (AUC = {roc_auc_grid:.2f})')
    plt.plot(fpr_random, tpr_random, color='blue', lw=2, label=f'Randomized Search ROC (AUC = {roc_auc_random:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ложно-положительная ставка')
    plt.ylabel('Истинно-положительная ставка')
    plt.title('Сравнение ROC-кривых')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_comparison_grid_random.png')
    plt.close()

    # 13.3 Визуализация Precision-Recall кривых для лучших моделей
    # Precision-Recall для Grid Search
    precision_grid, recall_grid, _ = precision_recall_curve(y_test, y_probs_grid)
    pr_auc_grid = auc(recall_grid, precision_grid)

    # Precision-Recall для Randomized Search
    precision_random, recall_random, _ = precision_recall_curve(y_test, y_probs_random)
    pr_auc_random = auc(recall_random, precision_random)

    # Визуализация Precision-Recall кривых
    plt.figure(figsize=(8, 6))
    plt.plot(recall_grid, precision_grid, color='purple', lw=2, label=f'Grid Search PR (AUC = {pr_auc_grid:.2f})')
    plt.plot(recall_random, precision_random, color='red', lw=2,
             label=f'Randomized Search PR (AUC = {pr_auc_random:.2f})')
    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title('Сравнение Precision-Recall кривых')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('precision_recall_comparison_grid_random.png')
    plt.close()

    print("\nВсе графики сохранены в текущей директории.")


if __name__ == "__main__":
    main()
