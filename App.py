# LIBRARY APP
import streamlit as st
from streamlit_option_menu import option_menu

# LIBRARY BERANDA
from PIL import Image

# LIBRARY PREDIKSI
from datetime import date
import numpy as np
import pandas as pd
from sklearn import datasets
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics


# JUDUL
st.header("Website Prediksi Saham PT.Telkom Indonesia")
st.write('\n')
st.write('\n')
st.write('\n')

# NAVBAR
selected = option_menu(
    menu_title=None,
    options=["Beranda", "Prediksi"],
    icons=["house", "book"],
    default_index = 0,
    orientation="horizontal",
)

st.write('\n')
st.write('\n')
st.write('\n')

# PROGRAM BERANDA
if selected == "Beranda":

    # st.subheader('PT. Telekomunikasi Indonesia')
    
    image = Image.open('assets/logo-telkom.png')
    st.image(image, caption=None, width=700, use_column_width='False', clamp=False, channels="RGB", output_format="auto")
    
    st.write('PT Telkom Indonesia (Persero) Tbk (Telkom) adalah Badan Usaha Milik Negara (BUMN) yang bergerak di bidang jasa layanan teknologi informasi dan komunikasi (TIK) dan jaringan telekomunikasi di Indonesia. Pemegang saham mayoritas Telkom adalah Pemerintah Republik Indonesia sebesar 52.09%, sedangkan 47.91% sisanya dikuasai oleh publik. Saham Telkom diperdagangkan di Bursa Efek Indonesia (BEI) dengan kode “TLKM” dan New York Stock Exchange (NYSE) dengan kode “TLK”.')
    st.write('Dalam upaya bertransformasi menjadi digital telecommunication company, TelkomGroup mengimplementasikan strategi bisnis dan operasional perusahaan yang berorientasi kepada pelanggan (customer-oriented). Transformasi tersebut akan membuat organisasi TelkomGroup menjadi lebih lean (ramping) dan agile (lincah) dalam beradaptasi dengan perubahan industri telekomunikasi yang berlangsung sangat cepat. Organisasi yang baru juga diharapkan dapat meningkatkan efisiensi dan efektivitas dalam menciptakan customer experience yang berkualitas.')
    st.write('Kegiatan usaha TelkomGroup bertumbuh dan berubah seiring dengan perkembangan teknologi, informasi dan digitalisasi, namun masih dalam koridor industri telekomunikasi dan informasi. Hal ini terlihat dari lini bisnis yang terus berkembang melengkapi legacy yang sudah ada sebelumnya.')
    st.write('Telkom mulai saat ini membagi bisnisnya menjadi 3 Digital Business Domain:')
    st.markdown('1. **Digital Connectivity:** Fiber to the x (FTTx), 5G, Software Defined Networking (SDN)/ Network Function Virtualization (NFV)/ Satellite')
    st.markdown('2. **Digital Platform:** Data Center, Cloud, Internet of Things (IoT), Big Data/ Artificial Intelligence (AI), Cybersecurity')
    st.markdown('3. **Digital Services:** Enterprise, Consumer')
    
    

# PROGRAM PREDIKSI
if selected == "Prediksi":

    #Scrapping data dari website yahoo finance
    START = "2017-08-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    
    stocks = ('TLKM.JK',)

    n_years = st.slider('Pilih berapa tahun untuk prediksi:', 1, 5)
    period = n_years * 365

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY,)
        data.reset_index(inplace=True)
        return data

    #Loading Data scrapping
    data_load_state = st.text('Memuat Data...') 
    data = load_data(stocks)
    data_load_state.text('Memuat Data, mohon tunggu sebentar..')

    st.subheader('Data Historis')
    st.write(data)
    
    st.subheader('Data Harga Realtime')
    st.write(data.tail())

    #Data Harga Realtime
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
            
    plot_raw_data()

    #Data Modelling
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    #Hasil Prediksi Prophet
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Data Prediksi')
    st.write(forecast.tail())

    st.write('Plot Prediksi untuk {n_years} Tahunan')
    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(
                xaxis_rangeselector_font_color='black',
                xaxis_rangeselector_activecolor='#7289DA',
                xaxis_rangeselector_bgcolor='#FFFFFF',
                )
    st.plotly_chart(fig1)
    st.subheader('Komponen Prediksi')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    st.subheader('Prediksi Grafik MAPE')
    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
    fig3 = plot_cross_validation_metric(df_cv, metric='mape')
    df_p = performance_metrics(df_cv)
    df_p.head()
    st.write(fig3)
        
    st.subheader("Tabel Cross Validation")
    df_p = performance_metrics(df_cv)
    df_p.head()
    st.code(df_p)

hide_st_style = """
    <style>
        ##MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """

st.markdown(hide_st_style, unsafe_allow_html=True)