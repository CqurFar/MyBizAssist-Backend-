import joblib
import numpy as np
import pandas as pd
from config import test_csv, train_csv


# Проверка наличия сохраненной модели
try:
    multi_target_clf = joblib.load("./model/multi_target_clf.pkl")
    vectorizer = joblib.load("./model/vectorizer.pkl")
    print("Модель и векторизатор успешно загружены.")
except FileNotFoundError:
    print("Модель catboost не найдена. Запуск обучения модели.")
    import mod_class  # Запускаем модель, если её нет


# Загрузка данных
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)


# Применение предобработки текста
def preprocess_text(text):
    import re
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    if isinstance(text, str):
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = text.lower()
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words("russian")]
        return " ".join(tokens)
    return ""


# Применение предобработки
test["text"] = test["text"].apply(preprocess_text)
test["combined"] = test["title"] + " " + test["text"]


# Векторизация
X_test = test["combined"]
X_test_tfidf = vectorizer.transform(X_test)


# Прогнозирование
y_pred = multi_target_clf.predict(X_test_tfidf)
y_pred = np.array(y_pred)


# Убираем лишнее измерение
if len(y_pred.shape) == 3 and y_pred.shape[0] == 1:
    y_pred = y_pred[0]

# Проверка формата
if y_pred.shape[0] == len(test) and y_pred.shape[1] == 2:
    pred_category, pred_class = y_pred[:, 0], y_pred[:, 1]
else:
    raise ValueError(f"Ошибка: форма y_pred ({y_pred.shape}) не соответствует ожиданиям.")


# Добавление предсказаний в DataFrame
test["category"] = pred_category
test["class"] = pred_class
final_df = pd.concat([train, test], ignore_index=True)


# Сохранение
final_df.to_csv("./data/data_final.csv", index=False, encoding="utf-8")
print("Результаты сохранены.")
