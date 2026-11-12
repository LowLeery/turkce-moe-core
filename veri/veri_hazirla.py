import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

veri_dosyasi = "veri/egitim_verisi.bin"
if os.path.exists(veri_dosyasi) and os.path.getsize(veri_dosyasi) > 100 * 1024 * 1024:
    print(f"KONTROL: {veri_dosyasi} zaten mevcut, indirme atlandi.")
    exit(0)

tokenizer = Tokenizer.from_file("tokenizer/turkce_kod_mat_tokenizer.json")
eos_id = tokenizer.token_to_id("<|endoftext|>")

def metin_al(ornek):
    for anahtar in ("text", "content", "code"):
        if anahtar in ornek and isinstance(ornek[anahtar], str):
            return ornek[anahtar]
    return ""

def veri_paketle():
    tum_tokenlar = []
    print("1/3 - Turkce veriler paketleniyor...")
    tr_veri = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    for i, ornek in enumerate(tr_veri):
        if i >= 50000: break
        tum_tokenlar.extend(tokenizer.encode(metin_al(ornek)).ids + [eos_id])

    print("2/3 - Python kod verileri paketleniyor...")
    kod_veri = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    for i, ornek in enumerate(kod_veri):
        if i >= 40000: break
        tum_tokenlar.extend(tokenizer.encode(metin_al(ornek)).ids + [eos_id])

    print("3/3 - Matematik verileri paketleniyor...")
    mat_veri = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    for i, ornek in enumerate(mat_veri):
        if i >= 25000: break
        tum_tokenlar.extend(tokenizer.encode(metin_al(ornek)).ids + [eos_id])

    dizi = np.array(tum_tokenlar, dtype=np.uint32)
    dizi.tofile(veri_dosyasi)
    print(f"Hazir! Toplam {len(dizi) / 1e6:.2f} Milyon token diske yazildi.")

if __name__ == "__main__":
    veri_paketle()
