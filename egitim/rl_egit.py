import sys, time, os, warnings, signal
warnings.filterwarnings("ignore")
sys.path.append(".")
import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
import gdown

from mimari.model import TurkceMoEModel, ModelAyarlari

print("🧠 RLVR Eğitimi Başlıyor (Zaman Aşımı Kalkanı Aktif)...", flush=True)

# 1. Cihaz ve Tokenizer Ayarları
cihaz = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("tokenizer/turkce_kod_mat_tokenizer.json")

sft_model_yolu = "turkce_moe_sft.pt"
if not os.path.exists(sft_model_yolu):
    DRIVE_LINKI = "https://drive.google.com/file/d/1UjISOSjJ3CIBD_P7SWpNnCaGvPTUA4T5/view?usp=sharing"
    dosya_id = DRIVE_LINKI.split("/d/")[1].split("/view")[0]
    print("SFT modeli Drive'dan indiriliyor...", flush=True)
    gdown.download(id=dosya_id, output=sft_model_yolu, quiet=False)

ayar = ModelAyarlari(max_seq_len=512)
model = TurkceMoEModel(ayar).to(cihaz)
model.load_state_dict(torch.load(sft_model_yolu, map_location=cihaz))
print("✅ SFT Temelli Model RLVR için Yüklendi!", flush=True)

gorevler = [
    {"soru": "Bir sayının çift olup olmadığını kontrol eden 'cift_mi' fonksiyonunu yaz.", "test_kodu": "assert cift_mi(4) == True\nassert cift_mi(5) == False\n"},
    {"soru": "Verilen iki sayının toplamını döndüren 'topla' fonksiyonunu yaz.", "test_kodu": "assert topla(2, 3) == 5\nassert topla(-1, 1) == 0\n"},
    {"soru": "Bir listenin içindeki eleman sayısını döndüren 'uzunluk' fonksiyonunu yaz.", "test_kodu": "assert uzunluk([1, 2, 3]) == 3\nassert uzunluk([]) == 0\n"}
]

# --- ZAMAN AŞIMI KALKANI (Sonsuz Döngü Engeli) ---
def zaman_asimi_yakalayici(signum, frame):
    raise TimeoutError("Sonsuz Döngü")

def kodu_dogrula(uretilen_kod, test_kodu):
    tam_kod = uretilen_kod + "\n" + test_kodu
    yerel_alan = {}
    
    # Koda en fazla 2 saniye çalışma süresi veriyoruz
    signal.signal(signal.SIGALRM, zaman_asimi_yakalayici)
    signal.alarm(2) 
    
    try:
        exec(tam_kod, {"__builtins__": __builtins__}, yerel_alan)
        signal.alarm(0) # Başarılıysa alarmı kapat
        return True
    except TimeoutError:
        print("⏳ Uyarı: Model sonsuz döngüye girdi, ceza veriliyor...", flush=True)
        return False
    except Exception:
        signal.alarm(0) # Diğer hatalar (Syntax Error vb.)
        return False
# ------------------------------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")

print("🚀 RLVR Döngüsü Başlatılıyor...", flush=True)
model.train()

for adim in range(1, 201):
    gorev = gorevler[(adim - 1) % len(gorevler)]
    prompt = f"<|user|>\n{gorev['soru']}\n<|assistant|>\n"
    girdi_ids = tokenizer.encode(prompt).ids
    girdi_tensor = torch.tensor([girdi_ids], dtype=torch.long).to(cihaz)
    
    uretilen_ids = []
    with torch.no_grad():
        for _ in range(120):
            logits, _ = model(girdi_tensor)
            son_token_logits = logits[0, -1, :] / 0.7
            olasiliklar = torch.nn.functional.softmax(son_token_logits, dim=-1)
            siradaki_token = torch.multinomial(olasiliklar, num_samples=1).item()
            if siradaki_token == tokenizer.token_to_id("<|endoftext|>"): break
            uretilen_ids.append(siradaki_token)
            yeni_tensor = torch.tensor([[siradaki_token]], dtype=torch.long).to(cihaz)
            girdi_tensor = torch.cat([girdi_tensor, yeni_tensor], dim=1)
            
    uretilen_metin = tokenizer.decode(uretilen_ids)
    basarili_mi = kodu_dogrula(uretilen_metin, gorev["test_kodu"])
    
    optimizer.zero_grad(set_to_none=True)
    if girdi_tensor.size(1) > 1:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits, _ = model(girdi_tensor[:, :-1])
            hedef = girdi_tensor[:, 1:]
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), hedef.reshape(-1), ignore_index=-150)
            
            # Başarılıysa -1 (Ödül), Hatalıysa +0.5 (Ceza)
            odul_carpani = -1.0 if basarili_mi else 0.5
            loss = loss * odul_carpani
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if adim % 5 == 0:
        durum = "🏆 BAŞARILI" if basarili_mi else "❌ HATA"
        print(f"Adım {adim}/200 | {durum}", flush=True)

rl_kayit_yolu = "/kaggle/working/turkce_moe_rlvr.pt"
torch.save(model.state_dict(), rl_kayit_yolu)
print(f"🔥 RLVR Tamamlandı! '{rl_kayit_yolu}' kaydedildi.", flush=True)
