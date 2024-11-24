import os
from config import url, links_new
from parcing import parse_gov_ru, process_documents_to_dataframe


def load_saved_links(file_path):
    """Загрузка ранее сохранённых ссылок."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return set(line.strip().split(". ")[-1] for line in file.readlines())
    return set()


def save_links(links, file_path):
    """Сохранение ссылок в файл."""
    with open(file_path, "w", encoding="utf-8") as file:
        for i, link in enumerate(sorted(links)):
            file.write(f"{i + 1:03d}. {link}\n")


def find_new_links(saved_links, current_links_file):
    """Поиск новых ссылок."""
    if not os.path.exists(current_links_file):
        return []

    with open(current_links_file, "r", encoding="utf-8") as file:
        current_links = set(line.strip().split(". ")[-1] for line in file.readlines())

    return list(current_links - saved_links)


def mstr(base_url, links, output_dir, output_csv):
    """Основная логика."""
    parse_gov_ru(url, output_dir)

    saved_links = load_saved_links(links)
    new_links = find_new_links(saved_links, links_new)

    if new_links:
        print(f"Найдено {len(new_links)} новых ссылок. Запускаем обработку...")

        all_links = saved_links.union(new_links)
        save_links(all_links, links)

        process_documents_to_dataframe(output_dir, output_csv)
        print("Обработка завершена.")
    else:
        print("Новых ссылок не найдено. Обработка не требуется.")
