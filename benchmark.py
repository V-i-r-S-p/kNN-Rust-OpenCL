import subprocess
import os
import csv
import re
import sys
from typing import List, Dict, Tuple
import time

# Конфигурация
RUST_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATE_SCRIPT = os.path.join(RUST_PROJECT_DIR, "generate_data.py")
RUST_BINARY = os.path.join(RUST_PROJECT_DIR, "target", "release", "knn")
RUST_BINARY = os.path.join(RUST_PROJECT_DIR, "target", "release", "knn")
RESULTS_FILE = os.path.join(RUST_PROJECT_DIR, "results.csv")

# Определяем команду Python в зависимости от ОС
PYTHON_CMD = sys.executable or "python3"

# Параметры для тестирования
N_VALUES = [5000, 10000, 50000, 100000]  # Количество точек
D_VALUES = [2, 4, 8]           # Размерность данных
W_VALUES = [None, 64, 128, 256]  # Work sizes (None для автоматического выбора)

def generate_data(n: int, d: int) -> Tuple[str, str]:
    """Генерирует тестовые данные и возвращает пути к файлам"""
    output_file = os.path.join(RUST_PROJECT_DIR, f"data_{n}_{d}.csv")
    test_file = os.path.join(RUST_PROJECT_DIR, f"data_{n}_{d}_test.csv")
    
    if not os.path.exists(output_file):
        print(f"Генерация данных: N={n}, D={d}")
        subprocess.run([
            PYTHON_CMD, GENERATE_SCRIPT,
            "-n", str(n),
            "-f", str(d),
            "-o", output_file
        ], check=True)
    
    return output_file, test_file
    
def cleanup_files(*files) -> None:
    """Удаляет указанные файлы, если они существуют"""
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"Удалён файл: {file}")
        except Exception as e:
            print(f"Ошибка при удалении {file}: {e}")

def run_rust_knn(data_file: str, test_point: str, use_opencl: bool, work_size: int = None) -> float:
    """Запускает Rust программу и возвращает время выполнения"""
    cmd = [
        RUST_BINARY,
        "-f", data_file,
        "-k", "5",
        "-p", test_point,
    ]
    
    if use_opencl:
        cmd.append("-o")
        if work_size is not None:
            cmd.extend(["-w", str(work_size)])
    
    print("Запуск: " + " ".join(cmd))
    start_time = time.time()
    result = subprocess.run(cmd, cwd=RUST_PROJECT_DIR, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Ошибка выполнения: {result.stderr}")
        return float('inf')
    
    # Парсим вывод Rust программы
    output = result.stdout
    time_match = re.search(r"Time elapsed: (\d+\.\d+)", output)
    if time_match:
        elapsed = float(time_match.group(1))
    
    return elapsed

def save_results(results: List[Dict]):
    """Сохраняет результаты в CSV файл"""
    file_exists = os.path.exists(RESULTS_FILE)
    
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["N", "D", "W", "TimeCPU", "TimeGPU"])
        
        if not file_exists:
            writer.writeheader()
        
        for row in results:
            writer.writerow(row)
    print(f"Результаты сохранены в {RESULTS_FILE}")

def main():
    # Собираем все комбинации параметров
    test_cases = []
    for n in N_VALUES:
        for d in D_VALUES:
            for w in W_VALUES:
                test_cases.append((n, d, w))
    
    results = []
    
    for n, d, w in test_cases:
        print(f"\n=== Тестирование N={n}, D={d}, W={w} ===")
        
        try:
            # Генерация данных
            data_file, test_file = generate_data(n, d)
            
            # Читаем тестовую точку из файла
            with open(test_file) as f:
                test_point = f.read().strip()
            
            # Запуск на CPU
            time_cpu = run_rust_knn(data_file, test_point, use_opencl=False)
            print(f"CPU время: {time_cpu:.4f} сек")
            
            # Запуск на GPU
            time_gpu = run_rust_knn(data_file, test_point, use_opencl=True, work_size=w)
            print(f"GPU время: {time_gpu:.4f} сек (work_size={w})")
            
            # Сохраняем результаты
            results.append({
                "N": n,
                "D": d,
                "W": w if w is not None else "auto",
                "TimeCPU": time_cpu,
                "TimeGPU": time_gpu
            })
        except Exception as e:
            print(f"Ошибка при тестировании N={n}, D={d}, W={w}: {str(e)}")
            
    
    # Сохранение всех результатов
    save_results(results)
    
    # Отчистка временных тестовых файлов
    for n in N_VALUES:
        for d in D_VALUES:
            cleanup_files(
                os.path.join(RUST_PROJECT_DIR, f"data_{n}_{d}.csv"),
                os.path.join(RUST_PROJECT_DIR, f"data_{n}_{d}_test.csv")
            )

if __name__ == "__main__":
    # Проверяем наличие Rust проекта
    if not os.path.exists(os.path.join(RUST_PROJECT_DIR, "Cargo.toml")):
        print("Ошибка: Не найден Rust проект в текущей директории")
        sys.exit(1)
    
    # Собираем Rust проект перед запуском
    print("Компиляция Rust проекта...")
    subprocess.run(["cargo", "build", "--release"], cwd=RUST_PROJECT_DIR, check=True)
    
    main()