import pandas as pd
import string

# Load the data from the CSV file
file_path = 'newdataset2.csv'
df = pd.read_csv(file_path)

# Function to clean the comment text
def clean_text(text):
    if isinstance(text, str):  # Check if text is a valid string
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        text = text.lower()
        return text
    return ''

# Apply the cleaning function to the 'Instagram Comment Text' column
df['text'] = df['text'].apply(clean_text)

# Handle missing values by filling them with an empty string
df['text'] = df['text'].fillna('')

# Create a new column with tokenized words
df['Words'] = df['text'].apply(lambda x: x.split())

# Display the updated DataFrame with the new 'Words' column
print(df[['Words', 'kategori']].head())

# Export the 'Words' column to an Excel file
df[['Words', 'kategori']].to_excel('Words2.xlsx', index=False)
