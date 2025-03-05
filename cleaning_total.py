import re
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Inisialisasi VADER untuk analisis sentimen
sia = SentimentIntensityAnalyzer()

# Membaca data dari file Excel
file_path = "datayangmaudicleaning.xlsx"
df_input = pd.read_excel(file_path)

# Pastikan ada kolom yang berisi teks (ganti 'KolomTeks' sesuai nama kolom di file Anda)
text_column = "text"  # Ganti dengan nama kolom yang sesuai
if text_column not in df_input.columns:
    raise ValueError(f"Kolom '{text_column}' tidak ditemukan dalam file.")

# Kata kunci untuk masing-masing pendukung
keywords = {
    "pendukung_anis": ["anis", "anies", "abah anies"],
    "pendukung_prabowo": ["prabowo", "bapak prabowo", "pak prabowo"],
    "pendukung_ganjar": ["ganjar", "pak ganjar", "rindu ganjar"]
}

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Menghapus karakter khusus
    text = text.lower().strip()  # Konversi ke huruf kecil
    return text

# Fungsi untuk menentukan kategori berdasarkan kata kunci
def classify_supporter(text):
    text = clean_text(text)
    for category, words in keywords.items():
        if any(word in text for word in words):
            return category
    return "tidak_jelas"

# Menganalisis sentimen dan menentukan kategori
results = []

for text in df_input[text_column].dropna():  # Menghindari nilai NaN
    cleaned_text = clean_text(text)
    sentiment_score = sia.polarity_scores(cleaned_text)['compound']
    category = classify_supporter(text)
    
    results.append({
        "Teks": text,
        "Kategori": category,
        "Sentimen": "Positif" if sentiment_score > 0.05 else "Negatif" if sentiment_score < -0.05 else "Netral"
    })

# Konversi ke DataFrame hasil
df_output = pd.DataFrame(results)

# Menyimpan hasil ke file Excel
output_path = "hasil_analisis_sentimen.xlsx"
df_output.to_excel(output_path, index=False)

# Menampilkan hasil klasifikasi
print(df_output)
