from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def veri_uretici():
    print("Veri akışları bağlanıyor...")
    
    # %45 Türkçe Veri (Wikipedia)
    tr_veri = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    for i, ornek in enumerate(tr_veri):
        if i >= 45000:
            break
        yield ornek["text"]

    # %35 Kod Verisi (Python)
    kod_veri = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
    for i, ornek in enumerate(kod_veri):
        if i >= 35000:
            break
        yield ornek["content"]

    # %20 Matematik Verisi
    mat_veri = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    for i, ornek in enumerate(mat_veri):
        if i >= 20000:
            break
        yield ornek["text"]

def tokenizer_egit():
    tokenizer = Tokenizer(models.BPE())
    
    # Sayıları tek tek basamaklara ayıran ve boşlukları koruyan ayar
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False)
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Özel sistem ve düşünme etiketleri
    ozel_tokenlar = [
        "<|endoftext|>", "<|pad|>", "<|user|>", "<|assistant|>",
        "<|dusunme_basla|>", "<|dusunme_bitir|>"
    ]

    egitici = trainers.BpeTrainer(
        vocab_size=65536,
        min_frequency=2,
        special_tokens=ozel_tokenlar,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print("Tokenizer eğitimi başlıyor...")
    tokenizer.train_from_iterator(veri_uretici(), trainer=egitici)
    tokenizer.save("tokenizer/turkce_kod_mat_tokenizer.json")
    print("Bitti! Dosya 'tokenizer/turkce_kod_mat_tokenizer.json' olarak kaydedildi.")

if __name__ == "__main__":
    tokenizer_egit()
