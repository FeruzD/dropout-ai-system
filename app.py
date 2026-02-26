import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# 1. Настройка страницы (строгий стиль)
st.set_page_config(page_title="DropOUT AI Expert", layout="wide", initial_sidebar_state="expanded")

# Кастомный CSS для профессиональных кнопок и вкладок
st.markdown("""
    <style>
    /* Основной фон страницы */
    .main { background-color: #0e1117; color: #ffffff; }
    
    /* Стилизация контейнера вкладок (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px; /* Расстояние между кнопками */
        background-color: transparent;
    }

    /* Стиль самих кнопок-вкладок */
    .stTabs [data-baseweb="tab"] {
        height: 60px; /* Увеличиваем высоту */
        min-width: 250px; /* Увеличиваем ширину */
        background-color: #1a1c24; /* Цвет неактивной кнопки */
        border-radius: 10px 10px 0px 0px; /* Закругленные углы сверху */
        color: #9ea0a5; /* Цвет текста */
        border: 1px solid #3d3d5c;
        transition: all 0.3s ease;
        
        /* ВАЖНО: Внутренние отступы, чтобы текст не прилипал */
        padding-left: 30px; 
        padding-right: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Эффект при наведении курсора */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #262936;
        color: #4facfe;
        border-color: #4facfe;
    }

    /* Стиль активной (выбранной) кнопки */
    .stTabs [aria-selected="true"] {
        background-color: #4facfe !important; /* Наш фирменный голубой */
        color: #ffffff !important;
        font-weight: bold;
        border-bottom: 4px solid #ffffff;
    }

    /* Центрирование текста внутри кнопки */
    .stTabs [data-baseweb="tab"] p {
        font-size: 18px;
        margin: 0px;
    }
            /* Стилизация полей ввода и поиска */
    .stTextInput input {
        height: 55px; /* Увеличиваем высоту поля поиска */
        font-size: 18px !important;
        border-radius: 10px !important;
        border: 2px solid #3d3d5c !important;
        background-color: #1a1c24 !important;
        color: white !important;
        padding-left: 15px !important;
    }

    /* Стилизация кнопок загрузки и скачивания */
    .stDownloadButton button, .stButton button {
        height: 60px !important; /* Большая высокая кнопка */
        width: 100% !important;
        font-size: 20px !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        transition: 0.3s all ease !important;
    }

    /* Заголовок над поиском, чтобы он был заметнее */
    .stTextInput label {
        font-size: 20px !important;
        color: #4facfe !important;
        margin-bottom: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Загрузка модели
@st.cache_resource
def load_model():
    try:
        return joblib.load('dropout_model_v2.pkl')
    except:
        return None

model = load_model()

# 3. Заголовок
st.title("🎓 AI Student Retention Analytics")
st.write("Samarqand mintaqaviy kasbiy ko'nikmalar markazi uchun intellektual tahlil va bashoratlash tizimi")
st.markdown("---")

if model is None:
    st.error("Model fayli (dropout_model_v2.pkl) topilmadi! Iltimos, avval 'step2_train_model_v2.py' faylini ishga tushiring.")
else:
    # Создаем вкладки
    tab1, tab2 = st.tabs(["👤 Individual Tahlil", "📂 Guruh Monitoringi (Excel/CSV)"])

    # Вкладка 1: Ручной ввод
    with tab1:
        st.subheader("📊 Talaba ko'rsatkichlarini kiriting")
        
        with st.form("input_form"):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("##### Личные данные")
                age = st.number_input("Yosh (Возраст)", 17, 50, 20)
                gender = st.selectbox("Jins (Пол)", [0, 1], format_func=lambda x: "Erkak" if x==0 else "Ayol")
                living = st.selectbox("Yashash joyi", [0, 1], format_func=lambda x: "Uyda" if x==0 else "Yotoqxona")
            
            with c2:
                st.markdown("##### Академические")
                attendance = st.slider("Davomat (%)", 0, 100, 85)
                gpa = st.slider("GPA (O'rtacha ball)", 1.0, 5.0, 3.8)
                failures = st.number_input("Qarzdorliklar soni", 0, 10, 0)
            
            with c3:
                st.markdown("##### Социальные")
                scholarship = st.selectbox("Stipendiya", [0, 1], format_func=lambda x: "Yo'q" if x==0 else "Ha")
                tuition = st.selectbox("Shartnoma to'lovi", [0, 1], format_func=lambda x: "Qarzdor" if x==0 else "To'langan")
                online = st.slider("LMS faolligi (soat/hafta)", 0, 40, 15)
            
            submit = st.form_submit_button("BASHORAT QILISH")

        if submit:
            # Подготовка данных для модели
            features = np.array([[age, gender, scholarship, tuition, living, attendance, gpa, failures, online]])
            prediction = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1] * 100

            # Визуализация Gauge (Спидометр)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "#4facfe"},
                    'steps': [
                        {'range': [0, 35], 'color': "#28a745"},
                        {'range': [35, 70], 'color': "#ffc107"},
                        {'range': [70, 100], 'color': "#dc3545"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}
                },
                title = {'text': "Dropout Xavfi (%)"}
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True)

            if prediction == 1:
                st.error(f"⚠️ DIQQAT: Talabada o'qishni tark etish xavfi YUQORI ({prob:.1f}%)")
            else:
                st.success(f"✅ Talabada o'qishni tark etish xavfi PAST ({prob:.1f}%)")

    # Вкладка 2: Загрузка файлов
    with tab2:
        st.subheader("📂 Guruh monitoringi va natijalarni eksport qilish")
        uploaded_file = st.file_uploader("Faylni tanlang (Excel yoki CSV)", type=["csv", "xlsx"])
        
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)
            
            # Колонки для ИИ (числовые)
            req_cols = ['Age', 'Gender', 'Scholarship', 'Tuition_Paid', 'Living_Status', 
                        'Attendance_Rate', 'GPA', 'Academic_Failures', 'Online_Activity']
            
            # Проверяем наличие колонки с именем (может называться Full_Name, ФИО или Имя)
            name_col = next((c for c in df_batch.columns if c.lower() in ['full_name', 'fio', 'ism', 'имя', 'фио']), None)

            if all(col in df_batch.columns for col in req_cols):
                # Очистка данных для модели
                X_batch = df_batch[req_cols].copy()
                for col in req_cols:
                    X_batch[col] = pd.to_numeric(X_batch[col], errors='coerce')
                X_batch = X_batch.fillna(0)
                
                # Прогноз
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)[:, 1] * 100
                
                # Добавляем результаты в таблицу
                df_batch['Risk (%)'] = np.round(probs, 1)
                df_batch['Status'] = ["Xavf ostida" if p == 1 else "Xavfsiz" for p in preds]
                
                # --- ПОИСК И ФИЛЬТРАЦИЯ ---
                search_query = st.text_input("🔍 Talabani ism-sharifi bo'yicha qidirish", "")
                
                display_df = df_batch.copy()
                if search_query and name_col:
                    display_df = display_df[display_df[name_col].astype(str).str.contains(search_query, case=False)]
                
                # Метрики
                m1, m2, m3 = st.columns(3)
                m1.metric("Jami talabalar", len(df_batch))
                m2.metric("Xavf ostidagilar", int(sum(preds)))
                m3.metric("O'rtacha xavf", f"{np.mean(probs):.1f}%")
                
                # Отображение таблицы (ФИО будет первой колонкой, если она есть)
                cols_to_show = ([name_col] if name_col else []) + ['Risk (%)', 'Status'] + req_cols
                st.dataframe(display_df[cols_to_show].style.background_gradient(subset=['Risk (%)'], cmap='Reds'), use_container_width=True)
                
                # --- ЭКСПОРТ В EXCEL ---
                st.markdown("### 📥 Hisobotni yuklab olish")
                
                # Конвертируем DataFrame в Excel в памяти
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_batch.to_excel(writer, index=False, sheet_name='Natijalar')
                
                st.download_button(
                    label="📄 Natijalarni Excel formatida yuklab olish",
                    data=output.getvalue(),
                    file_name="dropout_predictions_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error(f"Faylda kerakli ustunlar topilmadi! Talab etiladi: {', '.join(req_cols)}")

st.markdown("---")
st.info("Tizim magistrlik dissertatsiyasi doirasida ishlab chiqilgan. Muallif: Farrux Shomirzayev")