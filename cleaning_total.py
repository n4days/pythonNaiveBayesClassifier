import re
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Inisialisasi VADER untuk analisis sentimen
sia = SentimentIntensityAnalyzer()

# Data teks yang diberikan
text_data = [
    "Kelihatan sekali calon yg berkwalitas, punya kapasitas dan memiliki kompetensi, integritas dan layak sbg panutan",
    "Kasian pak ganjar",
    "Buktinya ganjar anis kmu kalah pilpres",
    "Anis cuma pinter ngomong waktu jadi gubernur DKI banyak ngibul tidak sesuai fakta. Tapi sekarang Kabar nya Anis banyak hutang nyungsep ke got jadi gembel politik 🎉😂",
    "Siapa yg menonton vidio ini setelah bapak prabowo terpilih❤",
    "aku cinta prabowo kiw kiw",
    "wow anis pintar ngmong ya tpi kerja nol mw ketawa tpi takut dosa",
    "Assalamualaikum abah anies yang kami hormati kami cintai  semoga sehat selalu  orang baik ya tetap orang baik",
    "prabowo menang anis menangissssssss mampus luuuuu"
]

# Kata kunci untuk masing-masing pendukung
keywords = {
    "pendukung_anis": ["anis", "anies", "abah anies"],
    "pendukung_prabowo": ["prabowo", "bapak prabowo", "pak prabowo"],
    "pendukung_ganjar": ["ganjar", "pak ganjar", "rindu ganjar"]
}

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Menghapus karakter khusus
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

for text in text_data:
    cleaned_text = clean_text(text)
    sentiment_score = sia.polarity_scores(cleaned_text)['compound']
    category = classify_supporter(text)
    
    results.append({
        "Teks": text,
        "Kategori": category,
        "Sentimen": "Positif" if sentiment_score > 0.05 else "Negatif" if sentiment_score < -0.05 else "Netral"
    })

# Konversi ke DataFrame
df = pd.DataFrame(results)

# Menampilkan hasil klasifikasi
print(df)
