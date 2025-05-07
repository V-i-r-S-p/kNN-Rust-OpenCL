import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import argparse
import os


def generate_clustered_data(num_points, num_features, num_clusters, output_file):
    # Генерация кластеризованных данных
    X, y = make_blobs(n_samples=num_points,
                      n_features=num_features,
                      centers=num_clusters,
                      cluster_std=1.0,
                      random_state=random.randrange(1, 1000))

    # Сохранение в CSV
    df = pd.DataFrame(X)
    df['label'] = y + 1  # Метки от 1 вместо 0
    df.to_csv(output_file, index=False, header=False)
    print(f"Данные сохранены в файл: {output_file}")

    # Генерация тестовой точки
    test_point = np.random.uniform(low=X.min(), high=X.max(), size=num_features)
    test_file = os.path.splitext(output_file)[0] + '_test.csv'
    pd.DataFrame([test_point]).to_csv(test_file, index=False, header=False)
    print(f"Тестовая точка сохранена в файл: {test_file}")

    # Визуализация для 2D случая
    if num_features == 2:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.scatter(test_point[0], test_point[1], c='red', marker='X', s=200, label='Тестовая точка')
        plt.title(f"Кластеризованные данные ({num_clusters} кластера)")
        plt.xlabel("Признак 1")
        plt.ylabel("Признак 2")
        plt.colorbar(scatter, label='Кластер')
        plt.legend()
        plt.grid(True)

        # Сохранение графика
        plot_file = output_file.replace('.csv', '.png')
        plt.savefig(plot_file)
        print(f"График сохранён в файл: {plot_file}")

    return test_point


if __name__ == "__main__":
    # Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(description='Генератор кластеризованных данных')
    parser.add_argument('-n', '--num_points', type=int, default=500,
                        help='Количество точек (по умолчанию: 500)')
    parser.add_argument('-f', '--num_features', type=int, default=2,
                        help='Количество признаков (по умолчанию: 2)')
    parser.add_argument('-c', '--num_clusters', type=int, default=3,
                        help='Количество кластеров (по умолчанию: 3)')
    parser.add_argument('-o', '--output', type=str, default='clustered_data.csv',
                        help='Имя выходного файла (по умолчанию: clustered_data.csv)')

    args = parser.parse_args()

    # Проверка введённых значений
    if args.num_points <= 0:
        print("Ошибка: количество точек должно быть положительным")
        exit(1)
    if args.num_features <= 0:
        print("Ошибка: количество признаков должно быть положительным")
        exit(1)
    if args.num_clusters <= 0:
        print("Ошибка: количество кластеров должно быть положительным")
        exit(1)

    print(f"\nГенерация данных со следующими параметрами:")
    print(f"Количество точек: {args.num_points}")
    print(f"Количество признаков: {args.num_features}")
    print(f"Количество кластеров: {args.num_clusters}")

    # Генерация данных
    test_point = generate_clustered_data(args.num_points, args.num_features,
                                         args.num_clusters, args.output)

    # Вывод команды для тестирования
    test_point_str = ",".join([f"{x:.2f}" for x in test_point])
    print("\nДля тестирования в kNN программе используйте команду:")
    print(f"cargo run -- -f {args.output} -k 5 -p \"{test_point_str}\" -l 0")