import sys, os
sys.path.append(".")
import torch
from tokenizers import Tokenizer
import gradio as gr

# Modeli indirebilmek icin gdown import ediyoruz
try:
    import gdown
except ImportError:
    os.system("pip install -q gdown")
    import gdown

from mimari.model import TurkceMoEModel, ModelAyarlari

# --- GUNCEL SFT MODEL LINKI ---
DRIVE_LINKI = "https://drive.google.com/file/d/1UjISOSjJ3CIBD_P7SWpNnCaGvPTUA4T5/view?usp=sharing"
# ------------------------------

cihaz = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Sistem basliyor... Kullanilan Cihaz: {cihaz}")

tokenizer = Tokenizer.from_file("tokenizer/turkce_kod_mat_tokenizer.json")
ayar = ModelAyarlari(max_seq_len=512)
model = TurkceMoEModel(ayar).to(cihaz)

# Modeli Kaggle'da veya Yerelde Bulma/Indirme Mantigi
sft_model_yolu = "/kaggle/working/turkce_moe_sft.pt"
if not os.path.exists(sft_model_yolu):
    sft_model_yolu = "turkce_moe_sft.pt"

if not os.path.exists(sft_model_yolu):
    print("Model yerelde bulunamadi. Hemen Drive'dan indiriliyor, lutfen bekle...")
    dosya_id = DRIVE_LINKI.split("/d/")[1].split("/view")[0]
    gdown.download(id=dosya_id, output=sft_model_yolu, quiet=False)

if os.path.exists(sft_model_yolu):
    model.load_state_dict(torch.load(sft_model_yolu, map_location=cihaz))
    model.eval()
    print("✅ SFT Modeli Basariyla Yuklendi ve Sohbete Hazir!")
else:
    print("❌ HATA: Model indirilemedi veya bulunamadi!")
    sys.exit(1)

# Sohbet Uretim Fonksiyonu
def cevap_uret(mesaj, gecmis, sicaklik, max_yeni_token):
    # Kullanici mesajini "user/assistant" ozel etiketleriyle sariyoruz
    formatli_girdi = f"<|user|>\n{mesaj}\n<|assistant|>\n"
    
    girdi_ids = tokenizer.encode(formatli_girdi).ids
    girdi_tensor = torch.tensor([girdi_ids], dtype=torch.long).to(cihaz)
    
    uretilen_ids = []
    
    with torch.no_grad():
        for _ in range(max_yeni_token):
            logits, _ = model(girdi_tensor)
            son_token_logits = logits[0, -1, :] / sicaklik
            
            olasiliklar = torch.nn.functional.softmax(son_token_logits, dim=-1)
            siradaki_token = torch.multinomial(olasiliklar, num_samples=1).item()
            
            if siradaki_token == tokenizer.token_to_id("<|endoftext|>"):
                break
                
            uretilen_ids.append(siradaki_token)
            
            yeni_token_tensor = torch.tensor([[siradaki_token]], dtype=torch.long).to(cihaz)
            girdi_tensor = torch.cat([girdi_tensor, yeni_token_tensor], dim=1)
            
            # Context tasmasini engelle
            if girdi_tensor.size(1) > ayar.max_seq_len:
                girdi_tensor = girdi_tensor[:, 1:]

    return tokenizer.decode(uretilen_ids)

# Gradio Arayuz Tasarimi
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Lise-AI Türkçe MoE 0.3B")
    gr.Markdown("Balıkesir'den çıkan sıfırdan eğitilmiş yapay zeka çekirdeği. (Canlı Test Arayüzü)")
    
    chatbot = gr.ChatInterface(
        fn=cevap_uret,
        chatbot=gr.Chatbot(height=450),
        textbox=gr.Textbox(placeholder="Bana yapay zekayı anlat veya kod yaz...", container=False, scale=7),
        additional_inputs=[
            gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature (Yaratıcılık)"),
            gr.Slider(minimum=10, maximum=300, value=150, step=10, label="Maksimum Uzunluk")
        ]
    )

# share=True sayesinde herkese acik link uretir
demo.launch(share=True)
