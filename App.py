# LIBRARY APP
# from turtle import position
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
st.markdown("<h1 style='text-align: center; color: black;'>Prediksi Saham <br> PT. Telekomunikasi Indonesia</h1>", unsafe_allow_html=True)
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
    
    st.markdown("<p style='text-align: justify; color: black;'>PT Telkom Indonesia (Persero) Tbk (Telkom) adalah Badan Usaha Milik Negara (BUMN) yang bergerak di bidang jasa layanan teknologi informasi dan komunikasi (TIK) dan jaringan telekomunikasi di Indonesia. Pemegang saham mayoritas Telkom adalah Pemerintah Republik Indonesia sebesar 52.09%, sedangkan 47.91% sisanya dikuasai oleh publik. Saham Telkom diperdagangkan di Bursa Efek Indonesia (BEI) dengan kode “TLKM” dan New York Stock Exchange (NYSE) dengan kode “TLK”.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; color: black;'>Dalam upaya bertransformasi menjadi digital telecommunication company, TelkomGroup mengimplementasikan strategi bisnis dan operasional perusahaan yang berorientasi kepada pelanggan (customer-oriented). Transformasi tersebut akan membuat organisasi TelkomGroup menjadi lebih lean (ramping) dan agile (lincah) dalam beradaptasi dengan perubahan industri telekomunikasi yang berlangsung sangat cepat. Organisasi yang baru juga diharapkan dapat meningkatkan efisiensi dan efektivitas dalam menciptakan customer experience yang berkualitas.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; color: black;'>Kegiatan usaha TelkomGroup bertumbuh dan berubah seiring dengan perkembangan teknologi, informasi dan digitalisasi, namun masih dalam koridor industri telekomunikasi dan informasi. Hal ini terlihat dari lini bisnis yang terus berkembang melengkapi legacy yang sudah ada sebelumnya.</p>", unsafe_allow_html=True)
    st.write('Telkom mulai saat ini membagi bisnisnya menjadi 3 Digital Business Domain:')
    st.markdown('1. **Digital Connectivity:** Fiber to the x (FTTx), 5G, Software Defined Networking (SDN)/ Network Function Virtualization (NFV)/ Satellite')
    st.markdown('2. **Digital Platform:** Data Center, Cloud, Internet of Things (IoT), Big Data/ Artificial Intelligence (AI), Cybersecurity')
    st.markdown('3. **Digital Services:** Enterprise, Consumer')
    
    st.write('\n')
    st.write('\n')
    st.write('\n')

    st.markdown("<h3 style='text-align: left; color: black;'>Sejarah PT. Telekomunikasi Indonesia</h3>", unsafe_allow_html=True)
    st.markdown("<table><tr><td>1882</td><td>Sebuah badan usaha swasta penyedia layanan pos dan telegrap dibentuk pada masa pemerintahan kolonial Belanda.</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>1906</td><td>Pemerintah Kolonial Belanda membentuk sebuah jawatan yang mengatur layanan pos dan telekomunikasi yang diberi nama Jawatan Pos, Telegrap dan (Post, Telegraph en Telephone Dienst/PTT).</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>1945</td><td>Proklamasi kemerdekaan Indonesia sebagai negara merdeka dan berdaulat, lepas dari pemerintahan Jepang.</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>1961</td><td>Status jawatan diubah menjadi Perusahaan Negara Pos dan Telekomunikasi (PN Postel). &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>1965</td><td>PN Postel dipecah menjadi Perusahaan Negara Pos dan Giro (PN Pos dan Giro), dan Perusahaan Negara Telekomunikasi (PN Telekomunikasi).</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>1974</td><td>PN Telekomunikasi disesuaikan menjadi Perusahaan Umum Telekomunikasi (Perumtel) yang menyelenggarakan jasa telekomunikasi nasional maupun internasional.</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>1980</td><td>PT Indonesian Satellite Corporation (Indosat) didirikan untuk menyelenggarakan jasa telekomunikasi internasional, terpisah dari Perumtel.</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>1989</td><td>Undang-undang No. 3 tahun 1989 tentang Telekomunikasi, tentang peran serta swasta dalam penyelenggaraan Telekomunikasi.</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>1991</td><td>Perumtel berubah bentuk menjadi Perusahaan Perseroan (Persero) Telekomunikasi Indonesia berdasarkan PP no. 25 tahun 1991.</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>1995</td><td>Penawaran Umum perdana saham TELKOM (Initial Public Offering) dilakukan pada tanggal 14 November 1995. sejak itu saham TELKOM tercatat dan diperdagangkan di Bursa Efek Jakarta (BEJ), Bursa Efek Surabaya (BES), New York Stock Exchange (NYSE) dan London Stock Exchange (LSE). Saham TELKOM juga diperdagangkan tanpa pencatatan (Public Offering Without Listing) di Tokyo Stock Exchange.</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>1999</td><td>Ditetapkan Undang-undang Nomor 36 Tahun 1999 tentang Telekomunikasi. Sejak tahun 1989, Pemerintah Indonesia melakukan deregulasi di sektor telekomunikasi dengan membuka kompetisi pasar bebas. Dengan demikian, Telkom tidak lagi memonopoli telekomunikasi Indonesia.</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>2001</td><td>Telkom membeli 35% saham Telkomsel dari Indosat sebagai bagian dari implementasi restrukturisasi industri jasa telekomunikasi di Indonesia yang ditandai dengan penghapusan kepemilikan bersama dan kepemilikan silang antara Telkom dan Indosat. Sejak bulan Agustus 2002 terjadi duopoli penyelenggaraan telekomunikasi lokal.</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>2009</td><td>Telkom meluncurkan 'New Telkom' ('Telkom baru') yang ditandai dengan penggantian identitas perusahaan.</td></tr></table>",unsafe_allow_html=True)
    st.markdown("<table><tr><td>2020</td><td>PT Telekomunikasi Indonesia (Persero) Tbk tercatat di Bursa Efek Indonesia dengan nama baru menjadi PT Telkom Indonesia (Persero) Tbk.</td></tr></table>",unsafe_allow_html=True)

    st.write('\n')
    st.write('\n')
    st.write('\n')

    st.markdown("<h3 style='text-align: center;'>Struktur Organisasi</h3>",unsafe_allow_html=True)
    image = Image.open('assets/struktur-telkom.png')
    st.image(image, caption=None, width=800, use_column_width='False', clamp=False, channels="RGB", output_format="auto")


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
    data = load_data(stocks)

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