import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Optimización de Campañas", layout="wide")

# Título principal
st.title(" Optimización de Campañas Publicitarias - Facebook Ads")
st.markdown("Sistema analítico para la optimización de inversión y estimación de retorno de inversión publicitaria (ROI).")

# Cargar datos de forma relativa (Compatible con Local y Colab)
@st.cache_data
def load_data():
    # Buscamos 'data.csv' en la misma carpeta donde se ejecuta la app
    df = pd.read_csv('data.csv')
    df['total_conversion'] = df['total_conversion'].fillna(0)
    df['approved_conversion'] = df['approved_conversion'].fillna(0)
    df['CTR'] = np.where(df['impressions'] > 0, df['clicks'] / df['impressions'], 0.0)
    df['CPC'] = np.where(df['clicks'] > 0, df['spent'] / df['clicks'], 0.0)
    df['ROI'] = np.where(df['spent'] > 0, (df['approved_conversion'] * 50) / df['spent'], 0.0)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ No se encontró el archivo 'data.csv' en la carpeta actual. Asegúrate de colocarlo en el mismo directorio que este archivo 'app.py'.")
    st.stop()

# Sidebar / Filtros
st.sidebar.header("Filtros de Campaña")
selected_campaign = st.sidebar.selectbox("Seleccionar Identificador de Campaña", options=['Todas'] + list(df['campaign_id'].unique()))

filtered_df = df if selected_campaign == 'Todas' else df[df['campaign_id'] == selected_campaign]

# Mapeo de pestañas
tab1, tab2, tab3 = st.tabs(["📊 Dashboard de Campañas", "💡 Recomendación de Inversión", "⚔️ Comparación de Canales"])

with tab1:
    st.subheader("Análisis del Rendimiento General")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Impresiones Totales", f"{filtered_df['impressions'].sum():,.0f}")
    col2.metric("Clicks Totales", f"{filtered_df['clicks'].sum():,.0f}")
    col3.metric("Inversión Total", f"${filtered_df['spent'].sum():,.2f}")
    col4.metric("Conversiones Aprobadas", f"{filtered_df['approved_conversion'].sum():,.0f}")

    # Gráfico de Impresiones vs Conversiones
    fig = px.scatter(filtered_df, x="spent", y="approved_conversion", size="clicks", color="age",
                     title="Conversiones Aprobadas vs Presupuesto Invertido",
                     labels={"spent": "Inversión ($)", "approved_conversion": "Conversiones Aprobadas"})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Simulador y Recomendación de Inversión")
    st.write("Utiliza el modelo predictivo optimizado para estimar conversiones basadas en tu configuración de inversión.")

    with st.form("simulador_form"):
        col1, col2 = st.columns(2)
        with col1:
            input_spent = st.number_input("Inversión Proyectada ($)", min_value=1.0, max_value=5000.0, value=100.0)
            input_clicks = st.number_input("Clicks Proyectados", min_value=1, max_value=1000, value=50)
            input_interest = st.slider("Código de Interés Principal (15 - 110)", min_value=15, max_value=110, value=25)
        with col2:
            input_impressions = st.number_input("Impresiones Proyectadas", min_value=100, max_value=1000000, value=10000)
            input_age = st.selectbox("Segmento de Edad", options=['30-34', '35-39', '40-44', '45-49'])
            input_gender = st.selectbox("Género Destinatario", options=['Femenino', 'Masculino'])

        submit = st.form_submit_button("Estimar Retorno")

        if submit:
            try:
                model = joblib.load('rf_campaign_model.pkl')

                # Preprocesar entradas con mapeo numérico consistente
                age_mapping = {'30-34': 1, '35-39': 2, '40-44': 3, '45-49': 4}
                age_val = age_mapping[input_age]
                gender_val = 1 if input_gender == 'Masculino' else 0
                ctr_val = input_clicks / input_impressions if input_impressions > 0 else 0.0
                cpc_val = input_spent / input_clicks if input_clicks > 0 else 0.0

                # Estructura del vector: ['interest1', 'interest2', 'interest3', 'impressions', 'clicks', 'spent', 'CTR', 'CPC', 'age_encoded', 'gender_encoded']
                features = [[input_interest, input_interest, input_interest, input_impressions, input_clicks, input_spent, ctr_val, cpc_val, age_val, gender_val]]

                pred_conversion = float(model.predict(features)[0])
                st.success(f"🎯 Conversiones Aprobadas Estimadas: **{pred_conversion:.2f}**")

                valor_estimado = pred_conversion * 50
                est_roi = (valor_estimado / input_spent) if input_spent > 0 else 0
                st.info(f"📈 ROI Estimado: **{est_roi:.2%}** (Basado en una utilidad de $50 por conversión)")
            except Exception as e:
                st.error(f"Error al procesar la predicción: {e}")

with tab3:
    st.subheader("Comparativa de Eficiencia por Segmentos (Canal de Edad)")
    segment_summary = df.groupby('age').agg({
        'spent': 'sum',
        'approved_conversion': 'sum'
    }).reset_index()
    segment_summary['Costo_Por_Conversion'] = segment_summary['spent'] / segment_summary['approved_conversion']

    fig_bar = px.bar(segment_summary, x='age', y='Costo_Por_Conversion',
                     title="Costo por Conversión Aprobada por Segmento de Edad (Menor es mejor)",
                     labels={'Costo_Por_Conversion': 'Costo por Conversión ($)', 'age': 'Grupo de Edad'},
                     color='age')
    st.plotly_chart(fig_bar, use_container_width=True)
