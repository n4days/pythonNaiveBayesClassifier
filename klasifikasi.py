import pandas as pd

# Load the Excel file
file_path = 'komentarprabowo.xlsx'
komentar_data = pd.read_excel(file_path)

# Function to classify comments
def classify_comment(comment):
    comment = str(comment).lower()
    if any(word in comment for word in ["bagus", "hebat", "suka"]):
        return "Suka"
    elif any(word in comment for word in ["buruk", "tidak suka", "jelek"]):
        return "Tidak Suka"
    else:
        return "Netral"

# Apply the classification function to each comment
komentar_data['Klasifikasi'] = komentar_data['komentar'].apply(classify_comment)

# Save the result to a new CSV file
output_path = 'hasil_klasifikasi_komentar.xlsx'
komentar_data.to_excel(output_path, index=False)

print(f"File Excel dengan klasifikasi telah disimpan sebagai: {output_path}")
