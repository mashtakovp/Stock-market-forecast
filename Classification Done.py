import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')
from textblob import TextBlob

dataset = pd.read_csv('AppleNewsStock.csv')
dataset_apple = dataset.copy()
dataset_apple.head(5)
dataset_apple.replace('NaN', pd.NA, inplace=True)
dataset_apple.dropna(subset=['News'], inplace=True)
dataset_apple.head(5)
#УДАЛЯЕМ ОДИНАКОВЫЕ НОВОСТИ
dataset_apple.drop_duplicates(subset ="News", keep = 'first', inplace = True)
dataset_apple.head(5)
#ПРИВОДИМ К НИЖНЕМУ РЕГИСТРУ
dataset_apple['News'] = dataset_apple['News'].str.lower()
dataset_apple.head(5)
dataset_apple.to_csv('apple_clear.csv')
#УДАЛИМ ЛИШНИЕ СИМВОЛЫ
dataset_apple['News'] = dataset_apple['News'].apply(lambda x: re.sub(r'\.{2,}', '.', x))  # Замена множественных точек на одну
dataset_apple['News'] = dataset_apple['News'].apply(lambda x: re.sub(r"''|\"\"|--", '', x))  # Удаление специфических символов
# Удаление стоп-слов
dataset_apple.head(5)
dataset_apple.to_csv('apple_test.csv')
#Стемминг
stemmer = PorterStemmer()

# Функция для стемминга текста
def stem_sentence(sentence):
    tokens = word_tokenize(sentence)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)
# Применение стемминга к каждой строке в столбце News
dataset_apple['News'] = dataset_apple['News'].apply(stem_sentence)

dataset_apple.head(5)
#ОПРЕДЕЛИМ СЕНТИМЕНТ
sia = SentimentIntensityAnalyzer()
def apply_sentiment(row):
    sentiment = sia.polarity_scores(row['News'])
    return pd.Series([sentiment['neg'], sentiment['neu'], sentiment['pos'], sentiment['compound']])
dataset_apple[['neg', 'neu', 'pos', 'compound']] = dataset_apple.apply(apply_sentiment, axis=1)
dataset_apple.head(5)
#TEXTBLOB
def Subjectivity(news):
  return TextBlob(news).sentiment.subjectivity

def Polarity(news):
  return TextBlob(news).sentiment.polarity

dataset_apple['subjectivity'] = dataset_apple['News'].apply(Subjectivity)
dataset_apple['polarity'] = dataset_apple['News'].apply(Polarity)
dataset_apple.head(5)
dataset_apple.to_csv('Full_data.csv')
#ЗАДАДИМ МЕТКИ КЛАССОВ
prices = dataset_apple[['Date', 'Adj Close']]
prices['Adj Close TMRW'] = prices['Adj Close'].shift(-1)
prices['Label'] = prices.apply(lambda x: 1 if (x['Adj Close TMRW']>= x['Adj Close']) else 0, axis =1)
prices = prices.drop(prices.index[-1]) #удалим последнюю строку, т.к. для нее нельзя определить класс
prices
#СОСТАВИМ ТАБЛИЦУ ДЛЯ ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ
first_dataset = dataset_apple.drop(['Adj Close', 'News'], axis=1)
first_dataset = first_dataset.drop(first_dataset.index[-1]) #удалим последнюю строку, т.к. для нее нельзя определить класс
full_data = pd.merge(first_dataset, prices, on='Date', how='right')
full_data
full_data.to_csv('for_classification.csv')
full_data['Date'] = pd.to_datetime(full_data['Date']) #преобразуем колонку с датой в числовое представление
full_data.set_index('Date', inplace=True)
#Разобьем датасет на обучающую и тестовую выборки
features = np.array(full_data.drop(['Label', 'Adj Close TMRW'], axis=1))
labels = np.array(full_data['Label'])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.35, random_state=2)
print(f"X train shape:{X_train.shape}")
print(f"y train shape:{y_train.shape}")
print(f"X test shape:{X_test.shape}")
print(f"y test shape:{y_test.shape}")
#Проведем нормализацию данных для их последующего использования в классификаторах
X_train = scale(X_train)
X_test = scale(X_test)
#Gridsearch и вывод результатов
classifiers = {
    'lr': LogisticRegression(),
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(),
    'mlp': MLPClassifier(max_iter=3000)
}

# Параметры для GridSearch
parameters = {
    'lr': {'C': np.logspace(-4, 4, 20), 'solver': ['liblinear']},
    'knn': {'n_neighbors': list(range(1, 11)), 'metric': ['euclidean', 'manhattan']},
    'rf': {'n_estimators': [10, 50, 100, 200], 'max_features': [None, 'sqrt', 'log2']},
    'mlp': {
        'hidden_layer_sizes': [(50,), (100,), (50,50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
}

best_models = {}

# Grid search для каждого классификатора
for name, classifier in classifiers.items():
    clf = GridSearchCV(classifier, parameters[name], cv=3, scoring='accuracy')
    clf.fit(X_train, y_train)
    best_models[name] = clf.best_estimator_
    print(f"{name} best params: {clf.best_params_}")
    print(f"{name} best score: {clf.best_score_}")

# Вывод матриц ошибок и classification report для лучших моделей
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{name} - Confusion Matrix:")
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f'Confusion Matrix for {name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred))