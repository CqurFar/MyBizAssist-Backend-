import os
import time
import subprocess
from search import mstr as search
from config import url, links, output_dir, output_csv
from parcing import parse_gov_ru, process_documents_to_dataframe


def check_links_file_exists():
    """Проверяет наличие файла links.txt"""
    return os.path.exists(links)


def start_parsing_and_processing():
    """Запускает парсинг и обработку данных"""
    print("Данные не найдены. Запуск стандартного парсинга...")
    parse_gov_ru(url, output_dir)
    process_documents_to_dataframe(output_dir, output_csv)
    print("Парсинг и обработка завершены. Данные сохранены.")


def run_models():
    """Запускает catboost и lda модели"""
    print("Запуск catboost...")
    subprocess.run(["python", "pred.py"])
    print("Работа классификатора завершена.")

    print("Запуск LDA...")
    subprocess.run(["python", "lda.py"])


def main():
    while True:
        if check_links_file_exists():
            print("Данные найдены. Выполняется проверка новых ссылок...")
            search(url, links, output_dir, output_csv)
        else:
            start_parsing_and_processing()

        run_models()

        print("Ожидание: 24 часа до следующей проверки...")
        time.sleep(86400)


if __name__ == "__main__":
    main()
