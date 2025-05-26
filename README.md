
# 📊 Проект: Бинарная классификация для предиктивного обслуживания оборудования

## 🧠 Описание проекта

Цель проекта — разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (`Target = 1`) или нет (`Target = 0`).  
Результаты оформлены в виде интерактивного Streamlit-приложения.

## 📁 Структура проекта

```
predictive_maintenance_project/
│
├── app.py                     # Главный файл Streamlit-приложения
├── analysis_and_model.py      # Страница анализа данных и моделей
├── presentation.py            # Презентация проекта (Streamlit Reveal Slides)
├── requirements.txt           # Зависимости проекта
├── README.md                  # Описание проекта
├── data/
│   └── predictive_maintenance.csv  # CSV с исходными данными (по желанию)
└── video/
    └── demo.mp4               # Видео-демонстрация работы (опционально)
```

## 📦 Установка и запуск

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/Suslik2005/VKR
   cd VKR
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Запустите приложение:
   ```bash
   streamlit run app.py
   ```

## 📈 Используемые модели

- Logistic Regression
- Random Forest
- XGBoost
- SVM

Для оценки применяются метрики: **Accuracy**, **ROC-AUC**, **Confusion Matrix**, **Classification Report** и **ROC-кривые**.

## 🖥️ Приложение

Возможности интерфейса:

- Загрузка данных
- Выбор модели
- Визуализация метрик и ROC-кривых
- Confusion Matrix для лучшей модели
- Предсказание по введённым данным

## 🗂️ Датасет

Используется датасет [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset)  
Он содержит 10 000 записей с различными параметрами оборудования и информацией об отказах.

## 🎥 Видео-демонстрация

Файл с демонстрацией находится в папке `video/`.

<video src="video/demo.mp4" controls width="100%"></video>

## 👤 Автор

Осипов Руслан, группа 5203  
Репозиторий проекта: [https://github.com/Suslik2005/VKR](https://github.com/Suslik2005/VKR)
