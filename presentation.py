import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    presentation_markdown = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    - Описание задачи: предсказать отказ оборудования.
    - Тип задачи: бинарная классификация (0 или 1).
    ---
    ## Этапы работы
    1. Загрузка и очистка данных.
    2. Предобработка.
    3. Обучение и сравнение моделей.
    4. Визуализация результатов.
    ---
    ## Streamlit-приложение
    - Анализ, обучение, предсказание.
    - Презентация.
    ---
    ## Заключение
    - Модель RandomForest показала наилучший результат.
    """

    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )