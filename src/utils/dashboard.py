import os
import streamlit as st
import pandas as pd
import altair as alt
import json
import io
import requests
from pathlib import Path

# Minimum rows required by the model (must match SEQUENCE_LEN in backend)
MIN_ROWS = 60

@st.cache_data
def get_report():
    base = Path(__file__).resolve().parents[2]
    path = base / 'models' / 'saved' / 'evaluation_report.json'
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)

@st.cache_data
def get_history():
    base = Path(__file__).resolve().parents[2]
    path = base / 'models' / 'saved' / 'training_history.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

# Model info

def page_model_info():
    st.title("Informações do Modelo")
    st.info("Esta página exibe as métricas de avaliação e o histórico de treinamento do modelo de predição de ações.")

    report_data = get_report()
    history_df = get_history()


    st.header("Relatório de Avaliação")
    st.subheader("Métricas do Modelo")

    cols = st.columns(3)
    cols[0].metric("MAE", f"{report_data['metrics']['mae']:.2f}")
    cols[1].metric("RMSE", f"{report_data['metrics']['rmse']:.2f}")
    cols[2].metric("R²", f"{report_data['metrics']['r2']:.2f}")
    cols[0].metric("MAPE", f"{report_data['metrics']['mape']:.2f}%")
    cols[1].metric("Acurácia Direcional", f"{report_data['metrics']['directional_accuracy']:.2f}%")

    st.markdown('---')

    col1, _, col2 = st.columns([1, 0.05, 1])
    with col1:
        st.subheader("Estatísticas de Erro")
        error_stats = report_data['error_statistics']
        error_df = pd.DataFrame.from_dict(error_stats, orient='index', columns=['Valor'])
        error_df = error_df.rename(index={
            "mean": "Média",
            "std": "Desvio Padrão",
            "min": "Mínimo",
            "max": "Máximo"
        })
        error_df.index.name = "Estatística"
        st.dataframe(error_df)

    with col2:
        st.subheader("Informações dos Dados")
        data_info = report_data['data_info'].copy()

        if 'price_range' in data_info and isinstance(data_info['price_range'], list) and len(data_info['price_range']) == 2:
            data_info['price_range'] = f"[{data_info['price_range'][0]:.3f}, {data_info['price_range'][1]:.3f}]"

        data_info_df = pd.DataFrame.from_dict(data_info, orient = 'index', columns = ['Valor'])
        data_info_df = data_info_df.rename(index = {"test_samples": "Amostras de Teste",
            "price_range": "Intervalo de Preço", "mean_price": "Preço Médio"})
        data_info_df.index.name = "Informação"
        data_info_df['Valor'] = data_info_df['Valor'].astype(str)
        st.dataframe(data_info_df)


    st.markdown('---')

    st.header("Histórico de Treinamento")

    history_df_long = history_df.reset_index().melt(
        'index',
        value_vars=['loss', 'mae', 'val_loss', 'val_mae'],
        var_name='Metric',
        value_name='Value'
    )
    history_df_long = history_df_long.rename(columns={'index': 'Época'})
    history_df_long['Metric'] = history_df_long['Metric'].replace({
        'loss': 'Perda (Treino)',
        'mae': 'MAE (Treino)',
        'val_loss': 'Perda (Validação)',
        'val_mae': 'MAE (Validação)'
    })

    chart = alt.Chart(history_df_long).mark_line(point=True).encode(
        x=alt.X('Época', axis=alt.Axis(title='Época')),
        y=alt.Y('Value', axis=alt.Axis(title='Valor')),
        color=alt.Color('Metric', title="Métrica"),
        tooltip=['Época', 'Metric', 'Value']
    ).interactive()

    st.altair_chart(chart, width='stretch')


# Inference

def api_call(symbol, days, data = None, kind=None, api_url='http://api-tech04.railway.internal:8080'):
    """
    Call to the prediction API.
    """

    # Resolve api_url: prefer explicit param, then environment variable, then localhost
    resolved = api_url or os.environ.get('API_URL') or 'api-tech04.railway.internal:8080'
    api_url = resolved.rstrip('/')
    if kind == 'stock_name':
        url = f"{api_url}/predict"
        # Backend expects `days_ahead` parameter name
        payload = {"symbol": symbol, "days_ahead": days}
    else:
        url = f"{api_url}/predict_custom"
        payload = {"data": data.to_json(), "days": days}

    try:
        response = requests.post(url, json=payload, timeout=20)
        # If response is error, show body to help debugging
        if response.status_code >= 400:
            try:
                body = response.text
            except Exception:
                body = '<no body>'
            st.error(f"API Error {response.status_code}: {body}")
            return None
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request failed: {e}")
        return None


def display_results(result):
    """Helper function to display prediction chart and data."""
    st.subheader(f"Predições para {result['symbol']}")

    df = pd.DataFrame({
        'date': pd.to_datetime(result['dates']),
        'prediction': result['predictions'],
        'lower_bound': result['confidence_interval']['lower'],
        'upper_bound': result['confidence_interval']['upper'],
    })

    base = alt.Chart(df).encode(
        x=alt.X('date', axis=alt.Axis(title='Data', format="%Y-%m-%d"))
    )

    band = base.mark_area(opacity=0.3, color='#57A44C').encode(
        y=alt.Y('lower_bound', axis=alt.Axis(title='Preço Previsto'), scale=alt.Scale(zero=False)),
        y2='upper_bound',
        tooltip=[
            alt.Tooltip('date', format="%Y-%m-%d", title="Data"),
            alt.Tooltip('lower_bound', format="$.2f", title="Limite Inferior"),
            alt.Tooltip('upper_bound', format="$.2f", title="Limite Superior")
        ]
    )

    line = base.mark_line(color='#E65133', point=True).encode(
        y=alt.Y('prediction', title="Predição", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('date', format="%Y-%m-%d", title="Data"),
            alt.Tooltip('prediction', format="$.2f", title="Predição")
        ]
    )

    # Combine chart
    chart = (band + line).interactive()
    st.altair_chart(chart, width='stretch')

    st.subheader("Dados Brutos da Predição")
    st.dataframe(df.set_index('date'))


def page_inference():
    st.title("Inferência do Modelo")
    st.info("Execute predições em tempo real usando a API do modelo treinado.")


    input_method = st.radio("Selecione o método de inferência:", (
    "Por Nome da Ação", "Por Dados de Ações Personalizados"), horizontal = True)

    st.markdown("---")

    stock_df = None


    if input_method == "Por Nome da Ação":
        st.header("Prever por Nome da Ação")
        symbol = st.selectbox("Selecione a ação", ["AAPL"], index=0)
        days_to_predict = st.number_input("Dias para Prever", min_value = 1, max_value = 30, value = 7)

        if st.button("Executar Predição", key = "btn_name"):
            with st.spinner("Executando inferência..."):
                result = api_call(symbol = symbol, days = days_to_predict, data = None, kind='stock_name')
                if result:
                    display_results(result)

    elif input_method == "Por Dados de Ações Personalizados":
        st.header("Prever por Dados de Ações Personalizados")
        with st.expander("No momento, o modelo foi configurado para aceitar as ações listadas abaixo"):
            st.markdown("- AAPL")
            st.warning("A tentativa de predição utilizando dados históricos de outras empresas pode trazer resultados ruins.")

        REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

        days_to_predict = st.number_input("Dias para Prever", min_value = 1, max_value = 30, value = 7)

        tab1, tab2 = st.tabs(["Carregar CSV", "Colar Dados"])

        with tab1:
            uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type = ["csv"])
            if uploaded_file:
                try:
                    stock_df = pd.read_csv(uploaded_file)
                    st.dataframe(stock_df.head())
                except Exception as e:
                    st.error(f"Erro ao ler o arquivo: {e}")
                    stock_df = None

        with tab2:
            st.info("Cole seus dados abaixo (formato CSV). A primeira linha deve ser o cabeçalho.")
            placeholder_csv = "Date,Open,High,Low,Close,Volume\n2024-10-01,150.0,152.0,149.0,151.0,100000\n..."
            pasted_data = st.text_area("Dados CSV colados", placeholder_csv, height = 150)
            if pasted_data and pasted_data != placeholder_csv:
                try:
                    stock_df = pd.read_csv(io.StringIO(pasted_data))
                    st.dataframe(stock_df.head())
                except Exception as e:
                    st.error(f"Erro ao processar os dados colados: {e}")
                    stock_df = None

        if st.button("Executar Predição", key = "btn_data"):
            if stock_df is not None:
                if all(col in stock_df.columns for col in REQUIRED_COLS):
                    # If dataset is smaller than SEQUENCE_LEN, warn but allow sending —
                    # backend will pad the history so a prediction can still be produced.
                    if len(stock_df) < MIN_ROWS:
                        st.warning(f"Aviso: o modelo usa uma janela de {MIN_ROWS} passos. Você enviou {len(stock_df)} linhas. O servidor irá completar os dados para permitir a predição (isso é apenas para demonstração).")
                    st.success("Colunas necessárias encontradas.")
                    with st.spinner("Executando inferência..."):
                        data_to_send = stock_df[REQUIRED_COLS]
                        result = api_call(symbol = "CUSTOM", days = days_to_predict, data = data_to_send, kind='stock_data')
                        if result:
                            display_results(result)
                else:
                    st.error(f"Dados faltando colunas obrigatórias. Deve conter: {REQUIRED_COLS}")
            else:
                st.warning("Por favor, carregue ou cole os dados para executar a predição.")



st.set_page_config(page_title = "Previsor de Ações", layout = "wide")
st.sidebar.title("Navegação")
page = st.sidebar.radio("Selecione uma Página", ["Informações do Modelo", "Inferência"])

if page == "Informações do Modelo":
    page_model_info()
elif page == "Inferência":
    page_inference()