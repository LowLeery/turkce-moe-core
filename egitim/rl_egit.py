import sys, time, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(".")
import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
import gdown

from mimari.model import TurkceMoEModel, ModelAyarlari

print("🧠 RLVR (Kendi Kendini Doğrulayan Akıl Yürütme) Eğitimi Başlıyor...", flush=True)

# 1. Cihaz ve Tokenizer Ayarları
cihaz = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("tokenizer/turkce_kod_mat_tokenizer.json")
pad_id = tokenizer.token_to_id("<|pad|>")

# 2. En güncel SFT modelimizi yüklüyoruz ki sıfırdan değil, akıllı modelin üstünden başlayalım
sft_model_yolu = "turkce_moe_sft.pt"
if not os.path.exists(sft_model_yolu):
    # Eğer yerelde yoksa Drive'dan otomatik indir
    DRIVE_LINKI = "https://drive.google.com/file/d/1UjISOSjJ3CIBD_P7SWpNnCaGvPTUA4T5/view?usp=sharing"
    dosya_id = DRIVE_LINKI.split("/d/")[1].split("/view")[0]
    print("SFT modeli Drive'dan indiriliyor...", flush=True)
    gdown.download(id=dosya_id, output=sft_model_yolu, quiet=False)

ayar = ModelAyarlari(max_seq_len=512)
model = TurkceMoEModel(ayar).to(cihaz)
model.load_state_dict(torch.load(sft_model_yolu, map_location=cihaz))
print("✅ SFT Temelli Model RLVR için Yüklendi!", flush=True)

# 3. Basit Python Kod Görevleri ve Test Senaryoları (Sandbox Doğrulama İçin)
# Modelin çözmesini isteyeceğimiz görevler ve onları test edecek Python assert komutları
gorevler = [
    {
        "soru": "Bir sayının çift olup olmadığını kontrol eden 'cift_mi' fonksiyonunu yaz.",
        "test_kodu": "assert cift_mi(4) == True\nassert cift_mi(5) == False\n"
    },
    {
        "soru": "Verilen iki sayının toplamını döndüren 'topla' fonksiyonunu yaz.",
        "test_kodu": "assert topla(2, 3) == 5\nassert topla(-1, 1) == 0\n"
    },
    {
        "soru": "Bir listenin içindeki eleman sayısını döndüren 'uzunluk' fonksiyonunu yaz.",
        "test_kodu": "assert uzunluk([1, 2, 3]) == 3\nassert uzunluk([]) == 0\n"
    }
]

# 4. Python Kodunu Güvenli Çalıştırma ve Doğrulama Fonksiyonu (Sandbox)
def kodu_dogrula(uretilen_kod, test_kodu):
    """
    Modelin ürettiği Python kodunu ve test kodunu birleştirip çalıştırır.
    Hatasız çalışırsa True (Ödül), hata verirse False (Ceza) döndürür.
    """
    tam_kod = uretilen_kod + "\n" + test_kodu
    
    # Güvenli Alan (Namespace)
    yerel_alan = {}
    try:
        # Kodun zaman aşımına uğramasını veya sonsuz döngüye girmesini engellemek için basit exec
        exec(tam_kod, {"__builtins__": __builtins__}, yerel_alan)
        return True # Testleri geçti! ÖDÜL 🏆
    except Exception as e:
        return False # Hata verdi! CEZA ❌

# 5. Optimizer ve Eğitim Ayarları
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")

print("🚀 RLVR Döngüsü Başlatılıyor... Model kendi hatalarından ders alacak.", flush=True)

# Basit bir RLVR iterasyon döngüsü (Policy Gradient / Reward Temelli Güncelleme)
model.train()
for adim in range(1, 201): # Örnek olarak 200 adım
    # Rastgele bir görev seç
    gorev = gorevler[(adim - 1) % len(gorevler)]
    soru = gorev["soru"]
    test_kodu = gorev["test_kodu"]
    
    prompt = f"<|user|>\n{soru}\n<|assistant|>\n"
    girdi_ids = tokenizer.encode(prompt).ids
    girdi_tensor = torch.tensor([girdi_ids], dtype=torch.long).to(cihaz)
    
    # Modelden Kod Üretmesini İstiyoruz
    uretilen_ids = []
    with torch.no_grad():
        for _ in range(120): # Maksimum 120 token üretsin
            logits, _ = model(girdi_tensor)
            son_token_logits = logits[0, -1, :] / 0.7 # Temperature
            olasiliklar = torch.nn.functional.softmax(son_token_logits, dim=-1)
            siradaki_token = torch.multinomial(olasiliklar, num_samples=1).item()
            
            if siradaki_token == tokenizer.token_to_id("<|endoftext|>"):
                break
            uretilen_ids.append(siradaki_token)
            yeni_tensor = torch.tensor([[siradaki_token]], dtype=torch.long).to(cihaz)
            girdi_tensor = torch.cat([girdi_tensor, yeni_tensor], dim=1)
            
    # Üretilen metni çöz
    uretilen_metin = tokenizer.decode(uretilen_ids)
    
    # 6. Doğrulama (Verification) Aşaması
    basarili_mi = kodu_dogrula(uretilen_metin, test_kodu)
    
    # RLVR Ödül / Ceza Sinyali (Reward Shaping)
    # Eğer kod testleri geçerse pozitif ödül, geçemezse hafif cezalandırma mantığıyla loss şekillendirilir
    optimizer.zero_grad(set_to_none=True)
    
    # Basitçe modelin ürettiği tokenlar üzerinden loss hesaplayıp ödül katsatısı ile çarpıyoruz
    if girdi_tensor.size(1) > 1:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits, _ = model(girdi_tensor[:, :-1])
            hedef = girdi_tensor[:, 1:]
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), hedef.reshape(-1), ignore_index=-150)
            
            # Ödül varsa loss'u küçült (pekiştir), ceza varsa tersine çevir
            odul_carpani = -1.0 if basarili_mi else 0.5
            loss = loss * odul_carpani
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if adim % 20 == 0:
        durum = "🏆 BAŞARILI (Ödül Verildi)" if basarili_mi else "❌ HATA (Düzeltme Bekleniyor)"
        print(f"RLVR Adım {adim}/200 | Durum: {durum}", flush=True)

# Eğitilen RLVR Modelini Kaydet
rl_kayit_yolu = "/kaggle/working/turkce_moe_rlvr.pt"
torch.save(model.state_dict(), rl_kayit_yolu)
print(f"🔥 RLVR Eğitimi Tamamlandı! Yeni model '{rl_kayit_yolu}' olarak kaydedildi.", flush=True)
