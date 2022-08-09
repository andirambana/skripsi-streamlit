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

# LIBRARY Navbar
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

    st.subheader('SEJARAH BINANCE COIN')
    
    # image = Image.open('logo.png')
    # st.image(image, caption=None, width=700, use_column_width='False', clamp=False, channels="RGB", output_format="auto")
    
    st.write('Diluncurkan pertama kali pada 26 Juni hingga 3 Juli 2017 dalam jaringan blockchain Ethereum, BNB memiliki harga penawarannya perdana sebesar 1 ETH untuk 2.700 BNB atau 1 BTC untuk 20.000 BNB.')
    st.write('Kendati ditawarkan perdana dalam ICO, aset digital ini tidak mengklaim sebagai produk investasi ataupun trading. Menurut pengembangnya, BNB adalah alat tukar yang digunakan untuk pembayaran, utamanya di platform yang berada dalam lingkup Binance, seperti binance.com, Binance DEX, Binance Chain, dan aplikasi di atas Binance Smart Chain.')
    st.write('Ketika ICO, sebanyak 10% atau 20 juta BNB dijual kepada angel investor. Apasih angel investor? Angel investor adalah sebutan untuk investor yang membiayai start up di awal pendiriannya. Lalu, sebanyak 40% atau 80 juta BNB diperuntukkan bagi tim, sementara sisanya atau 50% dari total suplai BNB atau sebanyak 100 juta koin lagi, dilepas ke pasar.')
    st.write('Adapun dana yang dihimpun dari ICO tersebut dipakai untuk membangun platform Binance dan menutup biaya operasional. Diketahui, sebanyak 30% dari total dana juga digunakan untuk membangun brand dan memasarkan BNB. Strategi ini terbukti sukses sebab terhitung 11 hari setelah penawaran perdananya, Binance telah memiliki platform sendiri.')
    
    
    st.subheader('SIAPAKAH PENDIRI BNB COIN?')

# PROGRAM PREDIKSI
if selected == "Prediksi":

        #Scrapping data dari yahoo finance
    START = "2017-05-21"
    TODAY = date.today().strftime("%Y-%m-%d")

    #Coin yang akan di input
    stocks = ('TLKM.JK',)
    # selected_stock = st.selectbox('Prediksi Data', stocks)

    #Lama durasi prediksi
    n_years = st.slider('Pilih berapa tahun untuk prediksi:', 1, 5)
    period = n_years * 365

    #cache data sehingga sistem tak perlu mengunduh data ulang
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    #Memuat data scrapping
    data_load_state = st.text('Memuat Data...') 
    data = load_data(stocks)
    data_load_state.text('Memuat Data, harap tunggu hingga selesai.')

    st.subheader('Data Harga ')
    st.write(data.tail())

    #Kode data harga sekarang
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
            
    plot_raw_data()

    #Modelling data
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    #Kode hasil prediksi prophet
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Prediksi Data')
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