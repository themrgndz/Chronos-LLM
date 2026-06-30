import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
import altair as alt
from collections import deque
from chronos import ChronosPipeline

# ==============================================================================
# 1. SAYFA AYARLARI VE MODELİN YÜKLENMESİ
# ==============================================================================
st.set_page_config(page_title="Canlı Trafik Analiz Paneli", layout="wide")

st.title("🧠 Kendi Kendini Optimize Eden Canlı Analiz & Forecasting Paneli")
st.markdown("RTX 4060 üzerinde **In-Context Learning** ile veri aktıkça kararlılığı artan ve başarı oranını anlık hesaplayan LLM sistemi.")

@st.cache_resource
def modeli_yukle():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-mini",
        device_map=device,
        torch_dtype=torch.bfloat16, # RTX 4060 için tam optimize hızlı veri tipi
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

st.subheader("💡 Çevrimiçi Adapte Olan Zaman Serisi Grafiği")
grafik_alani = st.empty()

st.subheader("🚨 Sistem Olay Logları (Anomali Kayıtları)")
log_alani = st.empty()

# ==============================================================================
# 4. CANLI DÖNGÜ VE ADAPTE OLAN HAFIZA MOTORU
# ==============================================================================
# Modelin beslendiği pencere sınırını kaldırıyoruz, zamanla büyüyerek akıllanacak
kayan_pencere = []
gecmis_df = pd.DataFrame(columns=["Zaman", "Değer", "Çizgi Tipi"])
kalici_gelecek_df = pd.DataFrame(columns=["Zaman", "Değer", "Çizgi Tipi"])
anomali_loglari = []
tahmin_hatalari = []

if st.button("Sistemi ve Canlı Analizi Ateşle"):
    
    for index, satir in df_kavsak.iterrows():
        zaman = satir["DateTime"]
        anlik_trafik = satir["Vehicles"]
        kayan_pencere.append(anlik_trafik)
        
        # Modelin kararlı çalışması için ilk 20 verinin birikmesini bekliyoruz
        if len(kayan_pencere) < 20:
            metrik_durum.metric("Sistem Durumu", "Veri Toplanıyor...", delta=f"{len(kayan_pencere)}/20", delta_color="off")
            time.sleep(0.01)
            continue
            
        # Bilgilendirme bandını güncelleyelim
        egitim_bandi.info(f"🤖 Boyutu: {len(kayan_pencere)} Saatlik Geçmiş Hafıza")

        # Chronos Anlık Tahmin (1 Adım)
        context = torch.tensor(kayan_pencere, dtype=torch.float32)
        tahmin_1 = pipeline.predict(context, 1)
        alt_esik, medyan, ust_esik = np.percentile(tahmin_1[0].numpy(), [5, 50, 95], axis=0)
        
        beklenen_deger = int(medyan[0])
        ust_limit = float(ust_esik[0])
        alt_limit = float(alt_esik[0])
        
        # --- 📊 DOĞRULUK ORANI HESABI (MAPE tabanlı) ---
        if anlik_trafik > 0:
            hata_orani = abs(anlik_trafik - beklenen_deger) / anlik_trafik
            tahmin_hatalari.append(hata_orani)
        
        # Son 30 adımdaki isabet yüzdesini hesapla
        guncel_mape = np.mean(tahmin_hatalari[-30:])
        dogruluk_yuzdesi = max(0.0, (1.0 - guncel_mape) * 100)
        
        # Anomali Kararı
        durum_yazisi = "✅ TEMİZ"
        if anlik_trafik > ust_limit:
            durum_yazisi = "🚨 YOĞUNLUK"
            anomali_loglari.append({"Zaman": zaman.strftime("%Y-%m-%d %H:%M:%S"), "Olay": "Aşırı Trafik Patlaması", "Değer": int(anlik_trafik)})
        elif anlik_trafik < alt_limit:
            durum_yazisi = "⚠️ TRAFİK ÇÖKÜŞÜ"
            anomali_loglari.append({"Zaman": zaman.strftime("%Y-%m-%d %H:%M:%S"), "Olay": "Sıra Dışı Trafik Düşüşü", "Değer": int(anlik_trafik)})
            
        # Metrik Kartlarını Güncelle
        metrik_trafik.metric("Anlık Trafik Hacmi", f"{anlik_trafik} Araç")
        metrik_beklenen.metric("Chronos Beklentisi", f"{beklenen_deger} Araç")
        metrik_durum.metric("Sistem Sağlığı", durum_yazisi)
        metrik_dogruluk.metric("Model Doğruluk Oranı", f"%{dogruluk_yuzdesi:.1f}", delta="Son 30 Adım")
        
        # --- KALICI FORECASTING: GELECEK 5 ADIM ---
        tahmin_5 = pipeline.predict(context, 5)
        gelecek_medyan = np.percentile(tahmin_5[0].numpy(), 50, axis=0)
        gelecek_zamanlar = pd.date_range(start=zaman, periods=6, freq='h')[1:]
        
        gelecek_listesi = []
        for gz, gm in zip(gelecek_zamanlar, gelecek_medyan):
            gelecek_listesi.append({
                "Zaman": gz, 
                "Değer": float(gm), 
                "Çizgi Tipi": "Gelecek Projeksiyonu (5s)"
            })
        yeni_gelecek_df = pd.DataFrame(gelecek_listesi)
        kalici_gelecek_df = pd.concat([kalici_gelecek_df, yeni_gelecek_df]).drop_duplicates(subset=["Zaman"], keep="last")
        
        # Havuzları hazırla ve birleştir
        yeni_gercek = pd.DataFrame({"Zaman": [zaman], "Değer": [float(anlik_trafik)], "Çizgi Tipi": ["Gerçek Trafik"]})
        yeni_beklenen = pd.DataFrame({"Zaman": [zaman], "Değer": [float(beklenen_deger)], "Çizgi Tipi": ["Anlık Beklenti"]})
        
        gecmis_df = pd.concat([gecmis_df, yeni_gercek, yeni_beklenen]).drop_duplicates(subset=["Zaman", "Çizgi Tipi"])
        
        grafik_havuzu = pd.concat([gecmis_df, kalici_gelecek_df])
        grafik_havuzu["Değer"] = grafik_havuzu["Değer"].astype(float)
        
        # --- ALTAIR GRAFİK ---
        chart = alt.Chart(grafik_havuzu).mark_line().encode(
            x=alt.X('Zaman:T', title='Zaman Ekseni'),
            y=alt.Y('Değer:Q', title='Araç / Trafik Sayısı', scale=alt.Scale(zero=False)),
            color=alt.Color('Çizgi Tipi:N', scale=alt.Scale(
                domain=['Gerçek Trafik', 'Anlık Beklenti', 'Gelecek Projeksiyonu (5s)'],
                range=['#FF4B4B', '#1F77B4', '#00CC96']
            )),
            strokeDash=alt.StrokeDash('Çizgi Tipi:N', scale=alt.Scale(
                domain=['Gerçek Trafik', 'Anlık Beklenti', 'Gelecek Projeksiyonu (5s)'],
                range=[[0, 0], [0, 0], [5, 5]]
            ))
        ).properties(height=450).interactive()
        
        grafik_alani.altair_chart(chart, width="stretch")
        
        if anomali_loglari:
            log_alani.dataframe(pd.DataFrame(anomali_loglari), width="stretch")
            
        time.sleep(0.4)