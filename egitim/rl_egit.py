import sys, time, os, warnings, signal, re
warnings.filterwarnings("ignore")
sys.path.append(".")
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from tokenizers import Tokenizer
import gdown

from mimari.model import TurkceMoEModel, ModelAyarlari

# --- DDP (ÇİFT GPU) BAŞLATMA ---
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    cihaz = f"cuda:{local_rank}"
    ana_islem = (local_rank == 0)
else:
    local_rank = 0
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    ana_islem = True

if ana_islem:
    print(f"🚀 ÇİFT GPU (DDP) RLVR AKTİF | Toplam GPU: {torch.cuda.device_count()}", flush=True)

tokenizer = Tokenizer.from_file("tokenizer/turkce_kod_mat_tokenizer.json")

sft_model_yolu = "turkce_moe_sft.pt"
if not os.path.exists(sft_model_yolu):
    if ana_islem:
        print("SFT modeli Drive'dan indiriliyor...", flush=True)
        DRIVE_LINKI = "https://drive.google.com/file/d/1UjISOSjJ3CIBD_P7SWpNnCaGvPTUA4T5/view?usp=sharing"
        dosya_id = DRIVE_LINKI.split("/d/")[1].split("/view")[0]
        gdown.download(id=dosya_id, output=sft_model_yolu, quiet=False)
    if ddp:
        dist.barrier()

ayar = ModelAyarlari(max_seq_len=512)
ham_model = TurkceMoEModel(ayar).to(cihaz)
ham_model.load_state_dict(torch.load(sft_model_yolu, map_location=cihaz))

model = DDP(ham_model, device_ids=[local_rank], find_unused_parameters=True) if ddp else ham_model

# 13 GÖREVLİK ZENGİN EĞİTİM PARKURU
gorevler = [
    {"soru": "Bir sayının çift olup olmadığını kontrol eden 'cift_mi' fonksiyonunu yaz.", "test_kodu": "assert cift_mi(4) == True\nassert cift_mi(5) == False"},
    {"soru": "Verilen iki sayının toplamını döndüren 'topla' fonksiyonunu yaz.", "test_kodu": "assert topla(2, 3) == 5\nassert topla(-1, 1) == 0"},
    {"soru": "Bir listenin içindeki eleman sayısını döndüren 'uzunluk' fonksiyonunu yaz.", "test_kodu": "assert uzunluk([1, 2, 3]) == 3\nassert uzunluk([]) == 0"},
    {"soru": "Verilen bir sayının faktöriyelini hesaplayan 'faktoriyel' fonksiyonunu yaz.", "test_kodu": "assert faktoriyel(5) == 120\nassert faktoriyel(0) == 1"},
    {"soru": "Bir metni tersine çeviren 'ters_cevir' fonksiyonunu yaz.", "test_kodu": "assert ters_cevir('lise') == 'esil'\nassert ters_cevir('yapay') == 'yapay'"},
    {"soru": "Verilen bir sayının asal olup olmadığını kontrol eden 'asal_mi' fonksiyonunu yaz.", "test_kodu": "assert asal_mi(7) == True\nassert asal_mi(10) == False"},
    {"soru": "Bir sayı listesindeki en büyük elemanı bulan 'en_buyuk' fonksiyonunu yaz.", "test_kodu": "assert en_buyuk([1, 5, 3]) == 5\nassert en_buyuk([-10, -2, -5]) == -2"},
    {"soru": "Bir kelimenin palindrom (tersten okunuşu aynı) olup olmadığını kontrol eden 'palindrom_mu' fonksiyonunu yaz.", "test_kodu": "assert palindrom_mu('radar') == True\nassert palindrom_mu('lise') == False"},
    {"soru": "Bir metindeki sesli harflerin sayısını bulan 'sesli_say' fonksiyonunu yaz.", "test_kodu": "assert sesli_say('merhaba') == 3\nassert sesli_say('xyz') == 0"},
    {"soru": "Bir sayı listesindeki sadece tek sayıları filtreleyip yeni bir liste döndüren 'tekleri_al' fonksiyonunu yaz.", "test_kodu": "assert tekleri_al([1, 2, 3, 4, 5]) == [1, 3, 5]\nassert tekleri_al([2, 4, 6]) == []"},
    {"soru": "Verilen iki kelimenin anagram (aynı harflerden oluşma) olup olmadığını kontrol eden 'anagram_mi' fonksiyonunu yaz.", "test_kodu": "assert anagram_mi('elma', 'amel') == True\nassert anagram_mi('lise', 'kedi') == False"},
    {"soru": "Derece cinsinden sıcaklığı Fahrenheit'a çeviren 'f_cevir' fonksiyonunu yaz.", "test_kodu": "assert f_cevir(0) == 32\nassert f_cevir(100) == 212"},
    {"soru": "Bir listedeki tüm sayıların ortalamasını bulan 'ortalama' fonksiyonunu yaz.", "test_kodu": "assert ortalama([10, 20, 30]) == 20.0\nassert ortalama([1, 2]) == 1.5"}
]

def zaman_asimi_yakalayici(signum, frame):
    raise TimeoutError("Sonsuz Dongu")

def kodu_dogrula(uretilen_metin, test_kodu):
    temiz_kod = uretilen_metin
    eslesme = re.search(r"```(?:python)?(.*?)```", uretilen_metin, re.DOTALL)
    if eslesme:
        temiz_kod = eslesme.group(1).strip()
    elif "<|assistant|>" in uretilen_metin:
        temiz_kod = uretilen_metin.split("<|assistant|>")[-1].strip()

    if not temiz_kod.strip(): 
        return False

    tam_kod = temiz_kod + "\n" + test_kodu
    yerel_alan = {}
    
    signal.signal(signal.SIGALRM, zaman_asimi_yakalayici)
    signal.alarm(2)
    try:
        exec(tam_kod, {"__builtins__": __builtins__}, yerel_alan)
        signal.alarm(0)
        return True 
    except Exception:
        signal.alarm(0)
        return False 

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")

# Kronometre için başlangıç zamanını kaydediyoruz
baslangic_zamani = time.time()

for adim in range(1, 5001):
    model.eval() 
    
    gorev_idx = (adim + local_rank) % len(gorevler) 
    gorev = gorevler[gorev_idx]
    
    prompt = f"<|user|>\n{gorev['soru']}\nLütfen sadece Python kodunu yaz, açıklama yapma.\n<|assistant|>\n"
    girdi_ids = tokenizer.encode(prompt).ids
    girdi_tensor = torch.tensor([girdi_ids], dtype=torch.long).to(cihaz)
    
    uretilen_ids = []
    with torch.no_grad():
        for _ in range(120):
            logits, _ = ham_model(girdi_tensor)
            son_token_logits = logits[0, -1, :] / 0.7
            olasiliklar = torch.nn.functional.softmax(son_token_logits, dim=-1)
            siradaki_token = torch.multinomial(olasiliklar, num_samples=1).item()
            if siradaki_token == tokenizer.token_to_id("<|endoftext|>"): break
            uretilen_ids.append(siradaki_token)
            yeni_tensor = torch.tensor([[siradaki_token]], dtype=torch.long).to(cihaz)
            girdi_tensor = torch.cat([girdi_tensor, yeni_tensor], dim=1)
            
    uretilen_metin = tokenizer.decode(uretilen_ids)
    basarili_mi = kodu_dogrula(uretilen_metin, gorev["test_kodu"])
    
    model.train() 
    optimizer.zero_grad(set_to_none=True)
    if girdi_tensor.size(1) > 1:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits, _ = model(girdi_tensor[:, :-1])
            hedef = girdi_tensor[:, 1:]
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), hedef.reshape(-1), ignore_index=-150)
            
            odul_carpani = -1.0 if basarili_mi else 0.2
            loss = loss * odul_carpani
            
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    # Log ekranı (Kronometre eklendi)
    if ana_islem and adim % 2 == 0:
        gecen_sure = time.time() - baslangic_zamani
        saat = int(gecen_sure // 3600)
        dakika = int((gecen_sure % 3600) // 60)
        saniye = int(gecen_sure % 60)
        zaman_yazisi = f"{saat:02d}:{dakika:02d}:{saniye:02d}"
        
        durum = "🏆 BAŞARILI (Ödül!)" if basarili_mi else "❌ HATA (Ceza)"
        print(f"[{zaman_yazisi}] DDP Adım {adim}/5000 | {durum}", flush=True)

    # Her 1000 adımda bir ara kayıt (Checkpoint) alıyoruz
    if ana_islem and adim % 1000 == 0:
        ara_kayit = f"/kaggle/working/turkce_moe_rlvr_{adim}.pt"
        torch.save(ham_model.state_dict(), ara_kayit)
        print(f"💾 EVRİM KAYDEDİLDİ: {ara_kayit}", flush=True)

if ana_islem:
    rl_kayit = "/kaggle/working/turkce_moe_rlvr_final.pt"
    torch.save(ham_model.state_dict(), rl_kayit)
    print(f"🔥 Çift GPU RLVR Eğitimi Tamamlandı! Model '{rl_kayit}' olarak kaydedildi.", flush=True)

if ddp:
    dist.destroy_process_group()
