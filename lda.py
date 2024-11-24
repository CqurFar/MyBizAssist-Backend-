import os
import re
import spacy
import pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser
from gensim.models.coherencemodel import CoherenceModel


# Загрузка данных
nlp = spacy.load("ru_core_news_sm")
df = pd.read_csv("./data/data_final.csv")
df["combined"] = df["text"].astype(str) + " " + df["title"].astype(str) + " " + df["subtitle"].astype(str)


# Дополнительные стоп-слова
CUSTOM_STOPWORDS = {
    "это", "который", "свой", "также", "мочь", "должен", "очень", "почему", "потому",
    "быть", "такой", "так", "весь", "если", "как", "кто", "что", "тот", "бы", "или",
    "нибудь", "уже", "еще", "тогда", "потом", "всего", "может", "сам", "каждый",
    "бывает", "даже", "раз", "нет", "вот", "только", "через", "когда", "мы", "вы",
    "они", "она", "оно", "себя", "этот", "этого", "этой", "эти", "этих", "ему", "ему",
    "её", "ими", "нам", "вам", "них", "ними", "наш", "ваш"
}


# Функции обработки текста
def separate_it(text):
    return re.sub(r"(?i)\bit\b", " it ", text)


def clean_text(text):
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.lemma_ not in CUSTOM_STOPWORDS
        and len(token) > 2
    ]
    return tokens


# Очистка и лемматизация
df["separated"] = df["combined"].apply(separate_it)
df["cleaned"] = df["separated"].apply(clean_text)
df["tokens"] = df["cleaned"].apply(preprocess_text)
df = df[df["tokens"].apply(len) > 3]


# Создание биграмм
bigram = Phrases(df["tokens"], min_count=3, threshold=50)
bigram_mod = Phraser(bigram)
df["bigram_tokens"] = df["tokens"].apply(lambda x: bigram_mod[x])


# Создание словаря и корпуса
dictionary = Dictionary(df["bigram_tokens"])
dictionary.filter_extremes(no_below=5, no_above=0.3)
corpus = [dictionary.doc2bow(text) for text in df["bigram_tokens"]]


# Функция для загрузки или обучения модели LDA
def load_or_train_lda_model(corpus, dictionary, model_path="./model/lda.model"):
    if os.path.exists(model_path):
        print("Модель успехно загружена.")
        lda_model = LdaModel.load(model_path)
    else:
        print("Модель LDA не найдена. Запуск обучения модели.")
        lda_model = LdaModel(corpus=corpus, num_topics=20, id2word=dictionary, passes=500, iterations=10000)
        lda_model.save(model_path)
        print("Модель LDA обучена и сохранена.")

    print("Работа LDA завершена.")
    return lda_model


# Загрузка или обучение модели LDA
lda_model = load_or_train_lda_model(corpus, dictionary)


# # Оценка модели с c_npmi и c_v
# coherence_model = CoherenceModel(model=lda_model, texts=df['bigram_tokens'], dictionary=dictionary, coherence='c_npmi', processes=1)
# npmi_coherence = coherence_model.get_coherence()
# print(f"NPMI Coherence: {npmi_coherence}")
#
# coherence_model_cv = CoherenceModel(model=lda_model, texts=df['bigram_tokens'], dictionary=dictionary, coherence='c_v', processes=1)
# cv_coherence = coherence_model_cv.get_coherence()
# print(f"C_v Coherence: {cv_coherence}")


# Прогнозирование темы для каждого документа
def get_dominant_topic(ldamodel, corpus):
    topics = ldamodel.get_document_topics(corpus)
    dominant_topics = [max(doc, key=lambda item: item[1])[0] for doc in topics]
    return dominant_topics


df["topic"] = get_dominant_topic(lda_model, corpus)
df["topic"] = df["topic"].apply(lambda x: f"topic_{x + 1:02d}")


# Сохранение
df.drop(columns=["combined", "separated"], inplace=True)
df.rename(columns={"id": "site"}, inplace=True)
df.loc[:105, "site"] = "sanctions"
df.loc[106:148, "site"] = "mob"
df.loc[149:171, "site"] = "sber"
df["id"] = range(1, len(df) + 1)

# Перестановка столбцов
columns = ["id"] + [col for col in df.columns if col != "id"]
df = df[columns]

# Экспорт
df.to_csv("./data/df_topics.csv", index=False, encoding="utf-16")
print("Результаты сохранены.")
