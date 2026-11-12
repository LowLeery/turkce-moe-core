from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/turkce_kod_mat_tokenizer.json")

ornekler = [
    "Muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine konuşma.",
    "Türkiye'nin yapay zekâ ekosistemi hızla gelişiyor.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Denklemin kökü: 1453 * 2026 = 2943778 olarak hesaplanır."
]

for metin in ornekler:
    kodlanmis = tokenizer.encode(metin)
    geri_cozulen = tokenizer.decode(kodlanmis.ids)
    
    print("\n" + "="*50)
    print(f"Orijinal Metin : {metin}")
    print(f"Token Sayısı   : {len(kodlanmis.ids)}")
    print(f"Token ID'leri  : {kodlanmis.ids[:10]}...")
    print(f"Geri Çözülen   : {geri_cozulen}")
    assert metin == geri_cozulen, "HATA: Metin bozuldu!"

print("\nBAŞARILI! Tüm metinler, Türkçe karakterler ve kodlar %100 kayıpsız çözüldü.")

