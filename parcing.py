import os
import time
import pandas as pd
from lxml import etree
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC


def parse_gov_ru(base_url, output_dir):
    """Парсинг сайта и извлечение данных"""
    os.makedirs(output_dir, exist_ok=True)

    # Инициализация драйвера
    def create_headless_driver():
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        return webdriver.Chrome(options=options)

    driver = create_headless_driver()
    driver.get(base_url)

    # Прокрутка страницы для загрузки контента
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # Масштабирование страницы
    driver.execute_script("document.body.style.zoom='10%'")
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(2)

    # Клик по вкладкам
    buttons = driver.find_elements(By.CSS_SELECTOR, "button.main-page__content-information-header-link")
    for button in buttons:
        ActionChains(driver).move_to_element(button).click(button).perform()
        time.sleep(1)

    # Сохранение main.html
    main_html_path = os.path.join(output_dir, "main.html")
    with open(main_html_path, "w", encoding="utf-8") as f:
        f.write(driver.page_source)

    # Извлечение ссылок
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")
    links = []

    articles = soup.find_all("a", class_="main-page__content-information-link")
    for article in articles:
        link = article.get("href")
        if link:
            links.append(f"http://government.ru{link}")

    # Сохранение ссылок
    links_path = os.path.join(output_dir, "links.txt")
    with open(links_path, "w", encoding="utf-8") as f:
        for i, link in enumerate(links):
            f.write(f"{i + 1:03d}. {link}\n")

    # Сохранение документов параллельно
    def download_page(driver, link, i):
        driver.execute_script(f"window.open('{link}', '_blank{i}');")
        driver.switch_to.window(driver.window_handles[-1])
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        html_content = driver.page_source
        doc_path = os.path.join(output_dir, f"gov_{i + 1:02d}.html")
        with open(doc_path, "w", encoding="utf-8") as file:
            file.write(html_content)
        time.sleep(1)
        driver.close()
        driver.switch_to.window(driver.window_handles[0])

    with ThreadPoolExecutor(max_workers=24) as executor:
        for i, link in enumerate(links):
            executor.submit(download_page, driver, link, i)

    driver.quit()


def process_documents_to_dataframe(output_dir, output_csv):
    """Обработка сохранённых документов и создание DataFrame"""

    # Извлечение данных из main.html
    main_html_path = os.path.join(output_dir, "main.html")
    with open(main_html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    class_indexes = {
        "People": 2,
        "Business": 3,
        "System": 4,
    }

    category_classes = {
        "main-page__content-information-link_blue": "Social",
        "main-page__content-information-link_orange": "Finance",
        "main-page__content-information-link_aqua": "Regulation",
        "main-page__content-information-link_pink": "Taxes",
        "main-page__content-information-link_green": "General",
    }

    data = []
    for category, index in class_indexes.items():
        container = soup.select_one(f"div.main-page__content-information:nth-of-type({index})")
        if not container:
            continue

        articles = container.find_all("a", class_="main-page__content-information-link")
        for article in articles:
            link = article.get("href")
            title = article.find("h4").get_text(strip=True) if article.find("h4") else None
            subtitle = article.find("p").get_text(strip=True) if article.find("p") else None

            link_class = article.get("class", [])
            category_color = "Unknown"
            for class_name in link_class:
                category_color = category_classes.get(class_name, "Unknown")
                if category_color != "Unknown":
                    break

            data.append(
                {
                    "id": len(data) + 1,
                    "link": f"http://government.ru{link}" if link else None,
                    "title": title,
                    "subtitle": subtitle,
                    "class": category,
                    "category": category_color,
                }
            )

    # Создание DataFrame
    df = pd.DataFrame(data)
    df["doc_id"] = df.index.map(lambda x: f"{x + 1:03d}")

    # Парсинг сохранённых документов
    def extract_document_data(file_path):
        tree = etree.parse(file_path, parser=etree.HTMLParser())

        text = []
        for i in range(1, 6):
            text_xpath = f"/html/body/div/section/div/div/div/div[1]/div[3]/div[1]/p[{i}]/text()"
            text_elements = tree.xpath(text_xpath)
            if text_elements:
                text.extend(text_elements)
            else:
                break

        date_xpath = "/html/body/div/section/div/div/div/div[1]/div[3]/div[2]/p/text()"
        date_elements = tree.xpath(date_xpath)
        date = " ".join(date_elements) if date_elements else None

        requirements_xpath = "/html/body/div/section/div/div/div/div[2]/div/div/p/text()"
        requirements_elements = tree.xpath(requirements_xpath)
        requirements = " ".join(requirements_elements) if requirements_elements else None

        approval = []
        for i in range(1, 6):
            approval_xpath = f"/html/body/div/section/div/div/div/div[3]/div/div/p[{i}]/text()"
            approval_elements = tree.xpath(approval_xpath)
            if approval_elements:
                approval.extend(approval_elements)
            else:
                break

        gov_links_xpath_1 = "/html/body/div/section/div/div/div/div[4]/div/div/a/@href"
        gov_links_1 = tree.xpath(gov_links_xpath_1)
        gov_links_xpath_2 = "/html/body/div/section/div/div/div/div[3]/div/div/a/@href"
        gov_links_2 = tree.xpath(gov_links_xpath_2)
        all_gov_links = gov_links_1 + gov_links_2

        # Логика для participate
        participate_xpath = "/html/body/div/section/div/div/div/div[3]/div/div//text()"
        participate_elements = tree.xpath(participate_xpath)
        participate = " ".join(participate_elements).strip() if participate_elements else None

        return {
            "text": " ".join(text),
            "date": date,
            "requirements": requirements,
            "approval": " ".join(approval),
            "gov": "; ".join(all_gov_links),
            "participate": participate,
        }

    # Обработка документов
    for doc_id in range(1, len(df) + 1):
        doc_file = f"gov_{doc_id:02d}.html"
        file_path = os.path.join(output_dir, doc_file)

        if os.path.exists(file_path):
            doc_data = extract_document_data(file_path)

            for column, value in doc_data.items():
                df.at[doc_id - 1, column] = value

    # Экспорт
    df.to_csv(output_csv, index=False, encoding="utf-8")
