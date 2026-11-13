import sys, os
sys.path.append(".")
import torch
from tokenizers import Tokenizer
from mimari.model import TurkceMoEModel, ModelAyarlari

cihaz = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("tokenizer/turkce_kod_mat_tokenizer.json")
ayar = ModelAyarlari(max_seq_len=512)
model = TurkceMoEModel(ayar).to(cihaz)

kayit_yolu = "/kaggle/working/turkce_moe_sft.pt" if os.path.exists("/kaggle/working/turkce_moe_sft.pt") else "checkpoints/turkce_moe_sft.pt"
model.load_state_dict(torch.load(kayit_yolu, map_location=cihaz))
model.eval()

def sor(kullanici_mesaji, max_yeni_token=150, tekrar_cezasi=1.25):
    prompt = f"<|user|>\n{kullanici_mesaji}\n<|assistant|>\n"
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=cihaz)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    
    with torch.no_grad():
        for _ in range(max_yeni_token):
            logits, _ = model(x[:, -ayar.max_seq_len:])
            son_logits = logits[:, -1, :].clone().squeeze(0)
            
            for token_id in set(next(iter(x.tolist()))):
                if son_logits[token_id] > 0:
                    son_logits[token_id] /= tekrar_cezasi
                else:
                    son_logits[token_id] *= tekrar_cezasi
                    
            son_logits = son_logits.unsqueeze(0) / 0.7
            probs = torch.softmax(son_logits, dim=-1)
            sonraki_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, sonraki_id), dim=1)
            if sonraki_id.item() == eos_id:
                break
                
    tam_cikti = tokenizer.decode(next(iter(x.tolist())))
    return tam_cikti.split("<|assistant|>")[-1].strip()

print("\n--- TEST 1: TURKCE GENEL BILGI ---", flush=True)
print("Soru: Yapay zeka nedir, kisaca acikla.", flush=True)
print("Cevap:", sor("Yapay zeka nedir, kisaca acikla."), flush=True)

print("\n--- TEST 2: PYTHON KOD YAZMA ---", flush=True)
print("Soru: Bir sayinin asal olup olmadigini bulan Python fonksiyonu yaz.", flush=True)
print("Cevap:", sor("Bir sayinin asal olup olmadigini bulan Python fonksiyonu yaz."), flush=True)
