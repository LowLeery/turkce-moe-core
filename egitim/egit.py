import sys, time, os
sys.path.append(".")
import numpy as np
import torch
from mimari.model import TurkceMoEModel, ModelAyarlari

cihaz = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Egitim baslatiliyor | Donanim: {cihaz.upper()}")

veri = np.memmap("veri/egitim_verisi.bin", dtype=np.uint32, mode="r")

def batch_al(batch_size, seq_len):
    ix = torch.randint(len(veri) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((veri[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((veri[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])
    return x.to(cihaz), y.to(cihaz)

if cihaz == "cuda":
    ayar = ModelAyarlari(max_seq_len=512)
else:
    ayar = ModelAyarlari(dim=128, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32, num_experts=4, expert_dim=128)

model = TurkceMoEModel(ayar).to(cihaz)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

os.makedirs("checkpoints", exist_ok=True)
batch_size = 8 if cihaz == "cuda" else 1
adim_sayisi = 5000 if cihaz == "cuda" else 3

baslangic = time.time()
for adim in range(1, adim_sayisi + 1):
    x, y = batch_al(batch_size, ayar.max_seq_len)
    with torch.amp.autocast(device_type=cihaz, dtype=torch.bfloat16, enabled=(cihaz=="cuda")):
        logits, loss = model(x, targets=y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if adim % 50 == 0 or cihaz == "cpu":
        print(f"Adim {adim}/{adim_sayisi} | Hata (Loss): {loss.item():.4f} | Sure: {time.time() - baslangic:.1f} sn")
    if cihaz == "cuda" and adim % 1000 == 0:
        torch.save(model.state_dict(), f"checkpoints/turkce_moe_adim_{adim}.pt")

if cihaz == "cuda":
    torch.save(model.state_dict(), "checkpoints/turkce_moe_final.pt")
