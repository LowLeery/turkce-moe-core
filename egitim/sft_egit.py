import sys, time, os, glob, warnings
warnings.filterwarnings("ignore")
sys.path.append(".")
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from tokenizers import Tokenizer
import gdown  # Drive'dan indirme islemi icin eklendi

from mimari.model import TurkceMoEModel, ModelAyarlari

# --- BURAYA GOOGLE DRIVE PAYLASIM LINKINI YAPISTIR ---
DRIVE_LINKI = "https://drive.google.com/file/d/1ODLYX08JM9JGt-LI1AJiMVHtvJHEa7wV/view?usp=sharing"
# ------------------------------------------------------

# DDP (Cift GPU Tam Guc) Baslatma
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
    print(f"DDP Tam Guc Egitim Basliyor | Aktif GPU Sayisi: {torch.cuda.device_count()}", flush=True)

tokenizer = Tokenizer.from_file("tokenizer/turkce_kod_mat_tokenizer.json")
pad_id = tokenizer.token_to_id("<|pad|>")

ornekler = []

# 1. Turkce Soru-Cevap Seti (Sutun isimlerini otomatik tespit eder)
if ana_islem:
    print("1/2 - Turkce Soru-Cevap verileri yukleniyor...", flush=True)
tr_sft = load_dataset("merve/turkish_instructions", split="train")
sutunlar = tr_sft.column_names

for satir in tr_sft:
    if len(ornekler) >= 15000:
        break
    degerler = [str(satir[c]).strip() for c in sutunlar if satir[c] is not None and str(satir[c]).strip() != ""]
    if len(degerler) >= 2:
        ilk = next(iter(degerler))
        son = degerler[-1]
        metin = f"<|user|>\n{ilk}\n<|assistant|>\n{son}<|endoftext|>"
        ids = tokenizer.encode(metin).ids
        if 15 < len(ids) <= 512:
            ornekler.append(ids)

# 2. Python Kod Seti (Dengeli oran icin 5.000 ornek)
if ana_islem:
    print(f"Basarili! {len(ornekler)} Turkce ornek alindi. Python verileri ekleniyor...", flush=True)
kod_sft = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
for i, satir in enumerate(kod_sft):
    if i >= 5000:
        break
    soru = str(satir.get("instruction") or "").strip()
    cevap = str(satir.get("output") or "").strip()
    if soru and cevap:
        metin = f"<|user|>\n{soru}\n<|assistant|>\n{cevap}<|endoftext|>"
        ids = tokenizer.encode(metin).ids
        if 15 < len(ids) <= 512:
            ornekler.append(ids)

if ana_islem:
    print(f"TOPLAM HAZIR VERI: {len(ornekler)} ornek! Iki GPU tam kapasite calisiyor...", flush=True)

def sft_batch_al(batch_size, max_len=512):
    ix = torch.randint(len(ornekler), (batch_size,))
    x_list, y_list = [], []
    for idx in ix:
        seq = ornekler[idx.item()]
        padded = seq + [pad_id] * (max_len - len(seq))
        x_list.append(torch.tensor(padded[:-1], dtype=torch.long))
        y_list.append(torch.tensor([t if t != pad_id else -100 for t in padded[1:]], dtype=torch.long))
    return torch.stack(x_list).to(cihaz), torch.stack(y_list).to(cihaz)

ayar = ModelAyarlari(max_seq_len=512)
ham_model = TurkceMoEModel(ayar).to(cihaz)

# Onceki SFT veya Temel Modeli otomatik bulur
sft_bul = glob.glob("/kaggle/input/**/turkce_moe_sft.pt", recursive=True)
temel_bul = glob.glob("/kaggle/input/**/turkce_moe_final.pt", recursive=True)
yuklenecek = next(iter(sft_bul), next(iter(temel_bul), "/kaggle/working/turkce_moe_final.pt"))

# === MODEL YUKLEME VE DRIVE'DAN INDIRME MANTIGI ===
if not os.path.exists(yuklenecek):
    indirme_yolu = "/kaggle/working/turkce_moe_final.pt"
    
    # Indirme ismini sadece ana islem (GPU 0) yapar
    if ana_islem:
        print("UYARI: Yerel model bulunamadi! Google Drive'dan indiriliyor...", flush=True)
        try:
            dosya_id = DRIVE_LINKI.split("/d/")[1].split("/view")[0]
            gdown.download(id=dosya_id, output=indirme_yolu, quiet=False)
        except Exception as e:
            print(f"HATA: Indirme basarisiz oldu. Detay: {e}")
            
    # Diger GPU(lar), ana islem indirmeyi bitirene kadar bekler
    if ddp:
        dist.barrier()
        
    yuklenecek = indirme_yolu

# Tüm GPU'lar dosyayi yuklemeyi dener
if os.path.exists(yuklenecek):
    ham_model.load_state_dict(torch.load(yuklenecek, map_location=cihaz))
    if ana_islem:
        print(f"BASARILI: '{yuklenecek}' yuklendi!", flush=True)
else:
    if ana_islem:
        print("HATA: Model dosyasi yerelde bulunamadi ve Drive'dan da indirilemedi! Egitim sifirdan basliyor.", flush=True)
# ===================================================

model = DDP(ham_model, device_ids=[local_rank], find_unused_parameters=True) if ddp else ham_model

batch_per_gpu = 6
grad_biriktirme = 2
toplam_adim = 1800

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.05, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=toplam_adim, eta_min=5e-6)
scaler = torch.amp.GradScaler("cuda")

sft_kayit = "/kaggle/working/turkce_moe_sft.pt" if os.path.exists("/kaggle/working") else "checkpoints/turkce_moe_sft.pt"
if ana_islem:
    os.makedirs(os.path.dirname(sft_kayit) or ".", exist_ok=True)
baslangic = time.time()

for adim in range(1, toplam_adim + 1):
    optimizer.zero_grad(set_to_none=True)
    toplam_loss = 0.0
    for step_idx in range(grad_biriktirme):
        x, y = sft_batch_al(batch_per_gpu, ayar.max_seq_len)
        if ddp:
            model.require_backward_grad_sync = (step_idx == grad_biriktirme - 1)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            loss = loss / grad_biriktirme
        scaler.scale(loss).backward()
        toplam_loss += loss.item()
        
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    
    if ana_islem and adim % 200 == 0:
        gecen = time.time() - baslangic
        print(f"DDP SFT Adim {adim}/{toplam_adim} | Loss: {toplam_loss:.4f} | Sure: {gecen/60:.1f} dk", flush=True)

if ana_islem:
    torch.save(ham_model.state_dict(), sft_kayit)
    print(f"TAMAMLANDI! Yeni model '{sft_kayit}' olarak kaydedildi.", flush=True)

if ddp:
    dist.destroy_process_group()
