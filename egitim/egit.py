import sys, time, os, shutil, glob, warnings
warnings.filterwarnings("ignore")
sys.path.append(".")
import numpy as np
import torch
import torch.nn as nn
from mimari.model import TurkceMoEModel, ModelAyarlari

cihaz = "cuda" if torch.cuda.is_available() else "cpu"
gpu_sayisi = torch.cuda.device_count()
print(f"Egitim Basliyor | Aktif GPU Sayisi: {gpu_sayisi}")

veri = np.memmap("veri/egitim_verisi.bin", dtype=np.uint32, mode="r")

def batch_al(batch_size, seq_len):
    ix = torch.randint(len(veri) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((veri[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((veri[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])
    return x.to(cihaz), y.to(cihaz)

def kullanilan_disk_gb():
    hedef = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    toplam, kullanilan, bos = shutil.disk_usage(hedef)
    return kullanilan / (1024 ** 3)

ayar = ModelAyarlari(max_seq_len=512)
ham_model = TurkceMoEModel(ayar).to(cihaz)

kayit_yolu = "/kaggle/working/turkce_moe_final.pt" if os.path.exists("/kaggle/working") else "checkpoints/turkce_moe_final.pt"
os.makedirs("checkpoints", exist_ok=True)

bulunanlar = glob.glob("/kaggle/input/**/turkce_moe_final.pt", recursive=True)
onceki_model = bulunanlar[0] if bulunanlar else kayit_yolu

if os.path.exists(onceki_model):
    ham_model.load_state_dict(torch.load(onceki_model, map_location=cihaz))
    print(f"BASARILI: {onceki_model} yuklendi, kaldigi yerden devam ediliyor!")
else:
    print("BILGI: Onceki model bulunamadi, sifirdan baslaniyor.")

model = nn.DataParallel(ham_model) if gpu_sayisi > 1 else ham_model

batch_size = 16 if gpu_sayisi > 1 else 8
maks_sure_saniye = 11.2 * 3600
maks_disk_gb = 17.5
toplam_tahmini_adim = 38000

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.1, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=toplam_tahmini_adim, eta_min=1e-5)
scaler = torch.amp.GradScaler("cuda")

baslangic = time.time()
adim = 0

while True:
    adim += 1
    x, y = batch_al(batch_size, ayar.max_seq_len)

with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
logits, loss = model(x, targets=y)
loss = loss.mean()
optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
scheduler.step()


gecen = time.time() - baslangic
if adim % 250 == 0:
guncel_lr = scheduler.get_last_lr()[0]
disk_gb = kullanilan_disk_gb()
print(f"Adim {adim} | Loss: {loss.item():.4f} | LR: {guncel_lr:.6f} | Disk: {disk_gb:.2f} GB | Sure: {gecen/60:.1f} dk")
if disk_gb >= maks_disk_gb:
torch.save(ham_model.state_dict(), kayit_yolu)
print(f"DISK SINIRI ({disk_gb:.2f} GB): Guvenli durduruldu ve kaydedildi!")
break
if adim % 1500 == 0:
torch.save(ham_model.state_dict(), kayit_yolu)
print(f"--> [Kayit Guncellendi] Adim {adim} tek dosya uzerine kaydedildi.")
if gecen >= maks_sure_saniye:
torch.save(ham_model.state_dict(), kayit_yolu)
print(f"Sure sinirina ulasildi ({adim} adim). Final model kaydedildi!")
break
