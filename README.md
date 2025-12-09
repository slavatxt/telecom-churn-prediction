# 🎯 Telecom Churn Prediction

Проект по предсказанию оттока клиентов телеком-компании с использованием машинного обучения.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 О проекте

Решение задачи [Advanced DLS Spring 2021](https://www.kaggle.com/competitions/advanced-dls-spring-2021) с Kaggle. Цель — предсказать, уйдет ли клиент от оператора связи на основе его характеристик и истории использования услуг.

### Основные результаты

- **Best Model:** Logistic Regression
- **ROC-AUC Score:** 0.7812
- **Accuracy:** 0.8061
- **F1-Score:** 0.6024

## 🚀 Быстрый старт

### Установка

```bash
# Клонируй репозиторий
git clone git@github.com:slavatxt/telecom-churn-prediction.git
cd telecom-churn-prediction

# Создай виртуальное окружение
python -m venv venv
source venv/bin/activate  # для macOS/Linux
# или
venv\Scripts\activate  # для Windows

# Установи зависимости
pip install -r requirements.txt
```

### Запуск анализа

```bash
# Открой Jupyter Notebook
jupyter notebook

# Запусти notebooks/01_churn_analysis.ipynb
```

### Обучение модели через скрипт

```bash
python src/models/train.py
```

### Создание предсказаний

```bash
python src/models/predict.py --model models/best_model.pkl --data data/raw/test.csv
```

## 📊 Данные

Датасет содержит информацию о клиентах телеком-компании:

- **Train:** 5,282 записей
- **Test:** 2,641 записей  
- **Features:** 19 признаков
- **Target:** Churn (0 - остался, 1 - ушел)

### Описание признаков

| Признак | Описание | Тип |
|---------|----------|-----|
| ClientPeriod | Количество месяцев обслуживания | Numeric |
| MonthlySpending | Ежемесячный платеж | Numeric |
| TotalSpent | Общая сумма платежей | Numeric |
| Sex | Пол клиента | Categorical |
| IsSeniorCitizen | Пожилой клиент (0/1) | Binary |
| HasPartner | Наличие партнера | Categorical |
| HasChild | Наличие детей | Categorical |
| HasPhoneService | Наличие телефонной услуги | Categorical |
| HasMultiplePhoneNumbers | Несколько номеров | Categorical |
| HasInternetService | Тип интернет-услуги | Categorical |
| HasOnlineSecurityService | Онлайн безопасность | Categorical |
| HasOnlineBackup | Онлайн бэкап | Categorical |
| HasDeviceProtection | Защита устройств | Categorical |
| HasTechSupportAccess | Техподдержка | Categorical |
| HasOnlineTV | Онлайн ТВ | Categorical |
| HasMovieSubscription | Подписка на фильмы | Categorical |
| HasContractPhone | Тип контракта | Categorical |
| IsBillingPaperless | Безбумажный биллинг | Binary |
| PaymentMethod | Способ оплаты | Categorical |

## 🔬 EDA и Insights

### Ключевые находки

1. **Дисбаланс классов:** 73.8% клиентов не уходят, 26.2% уходят
2. **Важные признаки:**
   - Тип контракта (месячный контракт → высокий churn rate ~43%)
   - Срок обслуживания (новые клиенты уходят чаще)
   - Тип интернета (Fiber optic → выше churn)
   - Способ оплаты (Electronic check → выше churn)
3. **Корреляции:**
   - TotalSpent и ClientPeriod сильно коррелируют (0.82)
   - MonthlySpending положительно коррелирует с Churn
   - ClientPeriod отрицательно коррелирует с Churn (-0.35)

Подробный анализ доступен в [notebooks/01_churn_analysis.ipynb](notebooks/01_churn_analysis.ipynb)

## 🤖 Модели

Протестированы следующие модели:

| Модель | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8061 | 0.6842 | 0.5379 | 0.6024 | 0.7812 |
| Decision Tree | 0.7845 | 0.6234 | 0.5145 | 0.5635 | 0.7456 |

**Финальная модель:** Logistic Regression с параметрами:
- `class_weight='balanced'` - для работы с дисбалансом классов
- `max_iter=1000`
- `random_state=42`

### Preprocessing Pipeline

1. Конвертация `TotalSpent` в numeric (был тип object)
2. Заполнение пропусков медианой
3. Label Encoding для категориальных признаков
4. StandardScaler для числовых признаков (только для Logistic Regression)

## 📁 Структура проекта

```
telecom-churn-prediction/
├── data/
│   ├── raw/                    # Исходные данные
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/              # Обработанные данные
├── notebooks/
│   └── 01_churn_analysis.ipynb # Полный анализ и моделирование
├── src/
│   ├── data/
│   │   └── preprocessing.py    # Препроцессинг данных
│   ├── features/
│   │   └── engineering.py      # Feature engineering
│   └── models/
│       ├── train.py            # Обучение моделей
│       └── predict.py          # Предсказания
├── models/                     # Сохраненные модели
├── submissions/                # Файлы для Kaggle
├── tests/                      # Тесты
├── requirements.txt            # Зависимости
├── setup.py                    # Установка пакета
└── README.md                   # Этот файл
```

## 🛠️ Технологии

- **Python 3.11** - язык программирования
- **pandas** - обработка данных
- **numpy** - численные вычисления
- **scikit-learn** - ML модели и метрики
- **matplotlib, seaborn** - визуализация
- **jupyter** - интерактивный анализ
- **pytest** - тестирование

## 📈 Roadmap

- [x] Exploratory Data Analysis
- [x] Baseline модели (Logistic Regression, Decision Tree)
- [ ] Продвинутые модели (Random Forest, XGBoost, LightGBM)
- [ ] Feature Engineering (создание новых признаков)
- [ ] Hyperparameter Tuning (Grid Search, Random Search)
- [ ] Cross-validation и ensemble методы
- [ ] Интерпретация моделей (SHAP, LIME)
- [ ] Deployment (Flask/FastAPI)

## 🤝 Contributing

Contributions are welcome! См. [CONTRIBUTING.md](CONTRIBUTING.md) для деталей.

### Workflow

1. Fork репозитория
2. Создай feature branch из `dev`: `git checkout -b feature/amazing-feature`
3. Commit изменения: `git commit -m 'feat: add amazing feature'`
4. Push в branch: `git push origin feature/amazing-feature`
5. Открой Pull Request в ветку `dev`

## 📝 License

Проект распространяется под лицензией MIT. См. [LICENSE](LICENSE) для деталей.

