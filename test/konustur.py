import sys, os
sys.path.append(".")
import torch
from tokenizers import Tokenizer
from mimari.model import TurkceMoEModel, ModelAyarlari

cihaz = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("tokenizer/turkce_kod_mat_tokenizer.json")
ayar = ModelAyarlari(max_seq_len=512)
model = TurkceMoEModel(ayar).to(cihaz)

kayit_yolu = "/kaggle/working/turkce_moe_final.pt" if os.path.exists("/kaggle/working") else "checkpoints/turkce_moe_final.pt"
model.load_state_dict(torch.load(kayit_yolu, map_location=cihaz))
model.eval()

def uret(baslangic_metni, max_yeni_token=100):
    ids = tokenizer.encode(baslangic_metni).ids
    x = torch.tensor([ids], dtype=torch.long, device=cihaz)
    with torch.no_grad():
        for _ in range(max_yeni_token):
            logits, _ = model(x[:, -ayar.max_seq_len:])
            son_logits = logits[:, -1, :] / 0.8
            probs = torch.softmax(son_logits, dim=-1)
            sonkiki_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, sonkiki_id), dim=1)
            if sonkiki_id.item() == torch.tensor([tokenizer.token_to_id("<|endoftext|>")]).item():
                break
    return tokenizer.decode(next(iter(x.tolist())))

print("\n--- TURKCE CIKTI ---")
print(uret("Türkiye'nin en önemli teknoloji"))
print("\n--- PYTHON KOD CIKTISI ---")
print(uret("def faktoriyel(n):\n"))
