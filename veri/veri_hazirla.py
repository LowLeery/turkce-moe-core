import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/turkce_kod_mat_tokenizer.json")
eos_id = tokenizer.token_to_id("<|endoftext|>")

def metin_al(ornek):
    for anahtar in ("text", "content", "code"):
        if anahtar in ornek and isinstance(ornek[anahtar], str):
            return ornek[anahtar]
    return ""

def veri_paketle():
    tum_tokenlar = []
    
    # 1. Türkçe Wikipedia (%45)
    print("Türkçe veriler tokenlara çevriliyor...")
    tr_veri = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    for i, ornek in enumerate(tr_veri):
        if i >= 20000: break
        ids = tokenizer.encode(metin_al(ornek)).ids
        tum_tokenlar.extend(ids + [eos_id])

    # 2. Python Kod Verisi (%35)
    print("Python kodları tokenlara çevriliyor...")
    kod_veri = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    for i, ornek in enumerate(kod_veri):
        if i >= 15000: break
        ids = tokenizer.encode(metin_al(ornek)).ids
        tum_tokenlar.extend(ids + [eos_id])

    # 3. Matematik Verisi (%20)
    print("Matematik verileri tokenlara çevriliyor...")
    mat_veri = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    for i, ornek in enumerate(mat_veri):
        if i >= 10000: break
        ids = tokenizer.encode(metin_al(ornek)).ids
        tum_tokenlar.extend(ids + [eos_id])

    dizi = np.array(tum_tokenlar, dtype=np.uint32)
    dizi.tofile("veri/egitim_verisi.bin")
    print(f"Tamamlandı! Toplam {len(dizi) / 1e6:.2f} Milyon token 'veri/egitim_verisi.bin' olarak kaydedildi.")

if __name__ == "__main__":
    veri_paketle()
