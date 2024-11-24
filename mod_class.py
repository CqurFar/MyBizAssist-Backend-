import re
import joblib
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from config import test_csv, train_csv
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from catboost import CatBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler, SMOTE


# Загрузка данных
train_data = pd.read_csv(test_csv)
test_data = pd.read_csv(train_csv)


# Предобработка текста
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = text.lower()
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stopwords.words("russian")
        ]
        return " ".join(tokens)
    return ""


train_data["text"] = train_data["text"].apply(preprocess_text)
test_data["text"] = test_data["text"].apply(preprocess_text)


# Объединение текста и заголовка
train_data["combined"] = train_data["title"] + " " + train_data["text"]
test_data["combined"] = test_data["title"] + " " + test_data["text"]


# Признаки и целевые переменные
X_train = train_data["combined"]
y_train = train_data[["category", "class"]]
X_test = test_data["combined"]


# Векторизация текста
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Автобалансировка классов
def balance_classes(X, y, apply_smote=False):
    X_resampled_list, y_resampled_list = [], []
    max_length = 0

    for i in range(y.shape[1]):
        if apply_smote:
            class_counts = y.iloc[:, i].value_counts()
            min_class_count = class_counts.min()
            if min_class_count > 1:
                smote = SMOTE(random_state=42, k_neighbors=min(min_class_count - 1, 5))
                X_resampled, y_resampled = smote.fit_resample(X, y.iloc[:, i])
            else:
                ros = RandomOverSampler(random_state=42)
                X_resampled, y_resampled = ros.fit_resample(X, y.iloc[:, i])
        else:
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y.iloc[:, i])

        max_length = max(max_length, len(y_resampled))

        X_resampled_list.append(X_resampled)
        y_resampled_list.append(y_resampled)

    X_resampled = np.hstack(X_resampled_list)

    y_resampled_padded = []
    for resampled in y_resampled_list:
        padding_size = max_length - len(resampled)
        if padding_size > 0:
            padded_resampled = np.pad(resampled, (0, padding_size), mode="constant", constant_values=-1)
        else:
            padded_resampled = resampled
        y_resampled_padded.append(padded_resampled.astype(str))

    for i, column in enumerate(y_resampled_padded):
        y_resampled_padded[i] = np.where(column == "-1", -1, column).astype(int)

    y_resampled = np.column_stack(y_resampled_padded)
    return X_resampled, y_resampled


# Параметры для модели
params = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "l2_leaf_reg": [1, 3, 5, 10, 15, 20],
    "iterations": [200, 500, 1000, 2000],
    "depth": [6, 8, 10, 12, 14],
    "border_count": [32, 64, 128, 254],
}

# Модель CatBoost
catboost_model = CatBoostClassifier(cat_features=[])


# Поиск гиперпараметров
random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=params,
    n_iter=5,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)


# Обучение модели
multi_target_clf = MultiOutputClassifier(random_search, n_jobs=-1)
multi_target_clf.fit(X_train_tfidf, y_train)


# Сохранение модели и векторизатора
joblib.dump(multi_target_clf, "multi_target_clf.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Модель и векторизатор успешно обучены и сохранены.")
