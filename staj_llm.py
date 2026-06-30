import pandas as pd
import numpy as np
import time
from collections import deque
import torch
from chronos import ChronosPipeline


print("Veri seti yükleniyor, hacı...")
df = pd.read_csv("traffic.csv")

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values('DateTime')

df_kavsak = df[df['Junction'] == 1].copy()
df_kavsak.set_index('DateTime', inplace=True)

print(f"Junction 1 için toplam veri sayısı: {len(df_kavsak)} satır.")

def canli_veri_akisi(dataframe):
    """
    Veri setindeki satırları sırayla, sanki canlı log geliyormuş gibi fırlatır.
    """
    for zaman, satir in dataframe.iterrows():
        veri_paketi = {
            "Zaman": zaman,
            "Trafik": satir["Vehicles"]
        }
        yield veri_paketi
        
kayan_pencere = deque(maxlen=10)

veri_motoru = canli_veri_akisi(df_kavsak)

print("\nCanlı akış simülasyonu başlıyor (Durdurmak için Ctrl+C veya Stop):\n")

try:
    # Test amaçlı ilk 15 verinin akışını simüle edelim
    for i in range(15):
        yeni_log = next(veri_motoru)
        
        kayan_pencere.append(yeni_log["Trafik"])
        
        print(f"[{yeni_log['Zaman']}] Anlık Gelen Trafik: {yeni_log['Trafik']} araç | Hafızadaki Son Durum (Pencere): {list(kayan_pencere)}")
        
        time.sleep(1)
        
except StopIteration:
    print("Veri bitti agam.")
    
# ==============================================================================
# MODELİN RTX 4060 ÜZERİNE YÜKLENMESI
# ==============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nModel Kontrolü -> Kod {device.upper()} üzerinde çalışacak.")

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-mini",
    device_map=device,
    torch_dtype=torch.bfloat16,
)
print("Chronos Başarıyla Hafızaya Alındı, Agam!\n")

# ==============================================================================
# CHRONOS ENTEGRASYONLU CANLI ANALİZ DÖNGÜSÜ
# ==============================================================================
kayan_pencere = deque(maxlen=100)
veri_motoru = canli_veri_akisi(df_kavsak)

print("Chronos canlı akışı tarıyor... (Durdurmak için Ctrl+C)")

try:
    for i in range(40):
        yeni_log = next(veri_motoru)
        anlik_trafik = yeni_log["Trafik"]
        kayan_pencere.append(anlik_trafik)
        
        # Modelin tahminde bulunabilmesi için pencerede en az 15-20 veri biriktiriyoruz
        if len(kayan_pencere) < 20:
            print(f"[{yeni_log['Zaman']}] Veri biriktiriliyor... ({len(kayan_pencere)}/20)")
            time.sleep(0.1)
            continue
            
        # --- CHRONOS VE NUMPY ANALİZ MOTORU ---
        # 1. Kayan pencereyi tensöre çevirip modelin emrine veriyoruz
        context = torch.tensor(list(kayan_pencere), dtype=torch.float32)
        
        # 2. Önümüzdeki 1 adımı (saati) tahmin
        tahmin = pipeline.predict(context, 1)
        
        # 3. NumPy devreye giriyor: Tahmin dağılımının %5, %50 ve %95'lik dilimlerini hesaplıyoruz
        tahmin_numpy = tahmin[0].numpy()
        alt_esik, medyan, ust_esik = np.percentile(tahmin_numpy, [5, 50, 95], axis=0)
        
        # 4. Anomali Karar Mekanizması
        durum = "✅ NORMAL"
        if anlik_trafik > ust_esik[0]:
            durum = "🚨 ANOMALİ (Aşırı Yoğunluk / DDoS Şüphesi)"
        elif anlik_trafik < alt_esik[0]:
            durum = "⚠️ ANOMALİ (Trafik Çöküşü / Sistem Hatası)"
            
        print(f"[{yeni_log['Zaman']}] Trafik: {anlik_trafik} | Beklenen: {int(medyan[0])} | Durum: {durum}")
        
        time.sleep(1) # İzlenebilirlik için 1 saniye bekleme
        
except StopIteration:
    print("Veri akışı bitti.")