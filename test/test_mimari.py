import sys
sys.path.append(".")
import torch
from tokenizers import Tokenizer
from mimari.model import TurkceMoEModel, ModelAyarlari

tokenizer = Tokenizer.from_file("tokenizer/turkce_kod_mat_tokenizer.json")
ayar = ModelAyarlari()
model = TurkceMoEModel(ayar)

toplam_parametre = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Toplam Model Parametresi: {toplam_parametre:.2f} Milyon")

# Kendi sözlüğümüzle gerçek bir Türkçe + Kod cümlesi verelim
test_metni = "def topla(a, b):\n    return a + b # Türkçe yorum"
girdi_ids = torch.tensor([tokenizer.encode(test_metni).ids])

logits, loss = model(girdi_ids, targets=girdi_ids)
print(f"Çıktı Matris Boyutu : {logits.shape}")
print(f"Başlangıç Hata (Loss): {loss.item():.4f}")
print("BAŞARILI! Mimari iskelet ile sözlüğümüz kusursuz eşleşti.")

