import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

def analysis_and_model_page():
    st.title("Анализ данных и модель")

    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Удаление ненужных столбцов
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

        # Очистка имён колонок от недопустимых символов
        data.columns = data.columns.str.replace(r'[\[\]<>]', '', regex=True)

        # Преобразование категориальной переменной Type
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Масштабирование числовых признаков
        numerical_features = ['Air temperature K', 'Process temperature K', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min']
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Деление на X и y
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']

        # Тренировочная и тестовая выборка
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Модели
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "SVM": SVC(kernel='linear', probability=True, random_state=42),
        }

        st.subheader("Результаты моделей")
        metrics = []
        best_model = None
        best_auc = 0
        best_name = ""
        best_pred = None

        # ROC-кривые
        plt.figure(figsize=(8, 6))
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)

            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

            metrics.append({"Model": name, "Accuracy": acc, "ROC AUC": roc_auc})

            if roc_auc > best_auc:
                best_model = model
                best_name = name
                best_pred = y_pred
                best_auc = roc_auc

            st.write(f"**{name}** - Accuracy: {acc:.2f}, ROC AUC: {roc_auc:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-кривые моделей")
        plt.legend()
        st.pyplot(plt)

        # Гистограмма сравнения метрик
        st.subheader("Сравнение моделей по Accuracy и ROC-AUC")
        metrics_df = pd.DataFrame(metrics)
        fig_bar, ax_bar = plt.subplots()
        metrics_df.set_index("Model")[["Accuracy", "ROC AUC"]].plot(kind="bar", ax=ax_bar)
        plt.title("Сравнение моделей")
        plt.ylabel("Значение")
        plt.ylim(0, 1)
        st.pyplot(fig_bar)

        # Confusion Matrix для лучшей модели
        st.subheader(f"Матрица ошибок (Confusion Matrix) — {best_name}")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, best_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel("Предсказано")
        ax_cm.set_ylabel("Истинное значение")
        st.pyplot(fig_cm)

        # Предсказание по новым данным
        st.subheader("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков:")

            type_input = st.selectbox("Тип продукта (L=0, M=1, H=2)", options=[0, 1, 2])
            air_temp = st.number_input("Температура воздуха (K)")
            process_temp = st.number_input("Температура процесса (K)")
            speed = st.number_input("Скорость вращения (rpm)")
            torque = st.number_input("Крутящий момент (Nm)")
            wear = st.number_input("Износ инструмента (min)")

            model_name = st.selectbox("Выберите модель для предсказания", options=list(models.keys()))
            submit_button = st.form_submit_button("Предсказать")

            if submit_button:
                input_df = pd.DataFrame([[type_input, air_temp, process_temp, speed, torque, wear]],
                                        columns=['Type', 'Air temperature K', 'Process temperature K', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min'])
                input_df[numerical_features] = scaler.transform(input_df[numerical_features])

                selected_model = models[model_name]
                prediction = selected_model.predict(input_df)[0]
                prediction_proba = selected_model.predict_proba(input_df)[0][1]

                st.markdown(f"### Результат предсказания: {'❌ Отказ' if prediction else '✅ Нет отказа'}")
                st.write(f"Вероятность отказа: **{prediction_proba:.2f}**")
