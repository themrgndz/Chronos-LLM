import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
import altair as alt
from chronos import ChronosPipeline

# ==============================================================================
# 1. SAYFA AYARLARI VE MODELİN YÜKLENMESİ
# ==============================================================================
st.set_page_config(page_title="Canlı Trafik Analiz Paneli", layout="wide")

st.title("🚀 Canlı Analiz & Forecasting Paneli")
st.markdown("**In-Context Learning** ile veri akışı kesilmeden çalışan LLM sistemi.")

@st.cache_resource
def modeli_yukle():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-mini",
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    return pipeline

pipeline = modeli_yukle()

# ==============================================================================
# 2. VERİ SETİN SÜRECİ
# ==============================================================================
@st.cache_data
def veri_hazirla():
    df = pd.read_csv("traffic.csv")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')
    df_kavsak = df[df['Junction'] == 1].copy()
    return df_kavsak

df_kavsak = veri_hazirla()

# ==============================================================================
# 🧠 OTURUM HAFIZASI (SESSION STATE) KURULUMU
# ==============================================================================
if "kayan_pencere" not in st.session_state:
    st.session_state.kayan_pencere = []
if "gecmis_df" not in st.session_state:
    st.session_state.gecmis_df = pd.DataFrame(columns=["Zaman", "Değer", "Çizgi Tipi"])
if "kalici_gelecek_df" not in st.session_state:
    st.session_state.kalici_gelecek_df = pd.DataFrame(columns=["Zaman", "Değer", "Çizgi Tipi"])
if "anomali_noktalari_df" not in st.session_state:
    st.session_state.anomali_noktalari_df = pd.DataFrame(columns=["Zaman", "Değer", "Tip"])
if "anomali_loglari" not in st.session_state:
    st.session_state.anomali_loglari = []
if "tahmin_hatalari" not in st.session_state:
    st.session_state.tahmin_hatalari = []
if "son_islenen_indeks" not in st.session_state:
    st.session_state.son_islenen_indeks = 0
if "sistem_aktif" not in st.session_state:
    st.session_state.sistem_aktif = False
if "secilen_anomali_zamani" not in st.session_state:
    st.session_state.secilen_anomali_zamani = None

# ==============================================================================
# 3. STREAMLIT ARAYÜZ ELEMANLARI
# ==============================================================================
kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)

with kpi_1:
    metrik_trafik = st.empty()
with kpi_2:
    metrik_beklenen = st.empty()
with kpi_3:
    metrik_durum = st.empty()
with kpi_4:
    metrik_dogruluk = st.empty()

egitim_bandi = st.empty()

st.subheader("📈 Çevrimiçi Adapte Olan Zaman Serisi Grafiği")

tum_çizgiler = ['Gerçek Trafik', 'Gelecek Projeksiyonu (1s)', 'Üst Sınır (%95 Güven)', 'Alt Sınır (%5 Güven)']
secilen_çizgiler = st.multiselect(
    "Grafikte Gösterilecek Çizgileri Seçin:",
    options=tum_çizgiler,
    default=tum_çizgiler
)

grafik_alani = st.empty()

st.subheader("🚨 Sistem Olay Logları (Anomali Kayıtları)")
st.caption("💡 İpucu: Aşağıdaki tablodan bir anomali seçtiğinizde grafik güncellenir. Canlı akış sırasında seçimi temizlemeniz önerilir.")
log_alani = st.empty()

# ==============================================================================
# 4. TETİKLEME BUTONLARI VE CANLI DÖNGÜ MOTORU
# ==============================================================================
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("Sistemi Ateşle / Devam Et"):
        st.session_state.sistem_aktif = True
        st.session_state.secilen_anomali_zamani = None  
with col_btn2:
    if st.button("Sistemi Durdur / Duraklat"):
        st.session_state.sistem_aktif = False

# Grafikte gösterilecek veriyi toparla
grafik_havuzu = pd.concat([st.session_state.gecmis_df, st.session_state.kalici_gelecek_df])
if not grafik_havuzu.empty:
    grafik_havuzu["Değer"] = grafik_havuzu["Değer"].astype(float)
    grafik_havuzu_filtreli = grafik_havuzu[grafik_havuzu["Çizgi Tipi"].isin(secilen_çizgiler)]
else:
    grafik_havuzu_filtreli = pd.DataFrame()

# CANLI AKIŞ DÖNGÜSÜ
if st.session_state.sistem_aktif:
    
    for index, satir in df_kavsak.iloc[st.session_state.son_islenen_indeks:].iterrows():
        
        if not st.session_state.sistem_aktif:
            break
            
        zaman = satir["DateTime"]
        anlik_trafik = satir["Vehicles"]
        st.session_state.kayan_pencere.append(anlik_trafik)
        st.session_state.son_islenen_indeks += 1
        
        if len(st.session_state.kayan_pencere) < 20:
            metrik_durum.metric("Sistem Durumu", "Veri Toplanıyor...", delta=f"{len(st.session_state.kayan_pencere)}/20", delta_color="off")
            time.sleep(0.01)
            continue
            
        egitim_bandi.info(f"🧠 Boyutu: {len(st.session_state.kayan_pencere)} Verilik Geçmiş Hafıza")

        # Chronos Anlık Tahmin
        context = torch.tensor(st.session_state.kayan_pencere, dtype=torch.float32)
        tahmin_1 = pipeline.predict(context, 1)
        alt_esik, medyan, ust_esik = np.percentile(tahmin_1[0].numpy(), [5, 50, 95], axis=0)
        
        beklenen_deger = int(medyan[0])
        ust_limit = float(ust_esik[0])
        alt_limit = float(alt_esik[0])
        
        if anlik_trafik > 0:
            hata_orani = abs(anlik_trafik - beklenen_deger) / anlik_trafik
            st.session_state.tahmin_hatalari.append(hata_orani)
        
        guncel_mape = np.mean(st.session_state.tahmin_hatalari[-100:])
        dogruluk_yuzdesi = max(0.0, (1.0 - guncel_mape) * 100)
        
        durum_yazisi = "🟢 TEMİZ"
        is_anomali = False
        
        if anlik_trafik > ust_limit:
            durum_yazisi = "🔴 ANORMAL YOĞUN"
            is_anomali = True
            st.session_state.anomali_loglari.append({"Zaman": zaman.strftime("%Y-%m-%d %H:%M:%S"), "Olay": "Aşırı Trafik Patlaması", "Değer": int(anlik_trafik)})
        elif anlik_trafik < alt_limit:
            durum_yazisi = "🔵 ANORMAL TENHA"
            is_anomali = True
            st.session_state.anomali_loglari.append({"Zaman": zaman.strftime("%Y-%m-%d %H:%M:%S"), "Olay": "Sıra Dışı Trafik Düşüşü", "Değer": int(anlik_trafik)})
            
        if is_anomali:
            yeni_anomali_noktasi = pd.DataFrame({"Zaman": [zaman], "Değer": [float(anlik_trafik)], "Tip": ["Anomali Noktası"]})
            st.session_state.anomali_noktalari_df = pd.concat([st.session_state.anomali_noktalari_df, yeni_anomali_noktasi]).drop_duplicates(subset=["Zaman"])
        
        # Metrik Kartları
        metrik_trafik.metric("Anlık Trafik Hacmi", f"{anlik_trafik} Araç")
        metrik_beklenen.metric("Chronos Beklentisi", f"{beklenen_deger} Araç")
        metrik_durum.metric("Sistem Sağlığı", durum_yazisi)
        metrik_dogruluk.metric("Model Doğruluk Oranı", f"%{dogruluk_yuzdesi:.1f}", delta="Son 100 Adım")
        
        # Gelecek 1 Adım Forecasting
        gelecek_zaman = zaman + pd.Timedelta(hours=1)
        gelecek_listesi = [{
            "Zaman": gelecek_zaman, 
            "Değer": float(beklenen_deger), 
            "Çizgi Tipi": "Gelecek Projeksiyonu (1s)"
        }]
        yeni_gelecek_df = pd.DataFrame(gelecek_listesi)
        st.session_state.kalici_gelecek_df = pd.concat([st.session_state.kalici_gelecek_df, yeni_gelecek_df]).drop_duplicates(subset=["Zaman"], keep="first")
        
        yeni_gercek = pd.DataFrame({"Zaman": [zaman], "Değer": [float(anlik_trafik)], "Çizgi Tipi": ["Gerçek Trafik"]})
        yeni_ust_limit = pd.DataFrame({"Zaman": [zaman], "Değer": [ust_limit], "Çizgi Tipi": ["Üst Sınır (%95 Güven)"]})
        yeni_alt_limit = pd.DataFrame({"Zaman": [zaman], "Değer": [alt_limit], "Çizgi Tipi": ["Alt Sınır (%5 Güven)"]})
        
        st.session_state.gecmis_df = pd.concat([st.session_state.gecmis_df, yeni_gercek, yeni_ust_limit, yeni_alt_limit]).drop_duplicates(subset=["Zaman", "Çizgi Tipi"])
        
        grafik_havuzu = pd.concat([st.session_state.gecmis_df, st.session_state.kalici_gelecek_df])
        grafik_havuzu["Değer"] = grafik_havuzu["Değer"].astype(float)
        grafik_havuzu_filtreli = grafik_havuzu[grafik_havuzu["Çizgi Tipi"].isin(secilen_çizgiler)]

        if st.session_state.anomali_loglari:
            log_df = pd.DataFrame(st.session_state.anomali_loglari)
            log_alani.dataframe(log_df, width="stretch", key=f"anomali_tablosu_{index}", on_select="ignore")

        # Grafiği Çiz
        ana_cizgiler = alt.Chart(grafik_havuzu_filtreli).mark_line().encode(
            x=alt.X('Zaman:T', title='Zaman Ekseni'),
            y=alt.Y('Değer:Q', title='Araç / Trafik Sayısı', scale=alt.Scale(zero=False)),
            color=alt.Color('Çizgi Tipi:N', scale=alt.Scale(
                domain=['Gerçek Trafik', 'Gelecek Projeksiyonu (1s)', 'Üst Sınır (%95 Güven)', 'Alt Sınır (%5 Güven)'],
                range=['#FF4B4B', '#00CC96', '#FFBB00', '#FFBB00']
            )),
            strokeDash=alt.StrokeDash('Çizgi Tipi:N', scale=alt.Scale(
                domain=['Gerçek Trafik', 'Gelecek Projeksiyonu (1s)', 'Üst Sınır (%95 Güven)', 'Alt Sınır (%5 Güven)'],
                range=[[0, 0], [4, 4], [3, 3], [3, 3]]
            ))
        )
        
        anomali_noktalari = alt.Chart(st.session_state.anomali_noktalari_df).mark_circle(size=100, color='#D62728').encode(
            x='Zaman:T', y='Değer:Q', tooltip=['Zaman:T', 'Değer:Q', 'Tip:N']
        )
        
        grafik_alani.altair_chart(alt.layer(ana_cizgiler, anomali_noktalari).properties(height=450).interactive(), width="stretch")
        time.sleep(0.4)

# ==============================================================================
# DÖNGÜ DIŞI / DURAKLAMA ANINDA ETKİLEŞİMLİ SEÇİM ALANI
# ==============================================================================
if not st.session_state.sistem_aktif and len(st.session_state.gecmis_df) > 0:
    
    if st.session_state.anomali_loglari:
        log_df = pd.DataFrame(st.session_state.anomali_loglari)
        secim_sonucu = log_alani.dataframe(
            log_df, width="stretch", on_select="rerun", selection_mode="single-row", key="sabit_anomali_tablosu"
        )
        if secim_sonucu and len(secim_sonucu.get("selection", {}).get("rows", [])) > 0:
            secilen_indeks = secim_sonucu["selection"]["rows"][0]
            st.session_state.secilen_anomali_zamani = log_df.iloc[secilen_indeks]["Zaman"]

    # Grafiği Çiz (Duraklama Modu)
    ana_cizgiler = alt.Chart(grafik_havuzu_filtreli).mark_line().encode(
        x=alt.X('Zaman:T'), y=alt.Y('Değer:Q', scale=alt.Scale(zero=False)), color='Çizgi Tipi:N'
    )
    
    if st.session_state.secilen_anomali_zamani:
        # HATA DÜZELTMESİ: Sadece saat-dakika değil, milisaniye bazında tam zaman damgası eşleşmesi yapılıyor 🚀
        secim_koşulu = f"time(datum.Zaman) == time(datetime('{st.session_state.secilen_anomali_zamani}'))"
        
        anomali_noktalari = alt.Chart(st.session_state.anomali_noktalari_df).mark_circle().encode(
            x='Zaman:T', y='Değer:Q',
            size=alt.condition(secim_koşulu, alt.value(300), alt.value(80)),
            color=alt.condition(secim_koşulu, alt.value('#FFD700'), alt.value('#D62728')),
            tooltip=['Zaman:T', 'Değer:Q', 'Tip:N']
        )
    else:
        anomali_noktalari = alt.Chart(st.session_state.anomali_noktalari_df).mark_circle(size=100, color='#D62728').encode(x='Zaman:T', y='Değer:Q')
        
    grafik_alani.altair_chart(alt.layer(ana_cizgiler, anomali_noktalari).properties(height=450).interactive(), width="stretch")

elif not st.session_state.sistem_aktif:
    log_alani.info("Sistem henüz başlatılmadı. Lütfen yukarıdaki butona basarak analizi ateşleyin.")