import os
import pandas as pd
from PIL import Image
import pytesseract
import requests
from nltk.tokenize import word_tokenize
from io import BytesIO
import nltk

nltk.download('punkt')

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 

posts_file = "output/buddhism_posts_incremental.csv"
image_dir = "output/images"
os.makedirs(image_dir, exist_ok=True)  
output_file = "output/tokenized_buddhism_media_texts.csv"

posts_df = pd.read_csv(posts_file)

media_urls = posts_df['media_url'].dropna().tolist() 

def download_image(url, local_dir):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        file_name = os.path.basename(url)
        local_path = os.path.join(local_dir, file_name)
        # Save the image
        with open(local_path, 'wb') as f:
            f.write(response.content)
        return local_path
    except Exception as e:
        print(f"Failed to download image at {url}: {e}")
        return None

def extract_text_from_image(local_path):
    try:
        img = Image.open(local_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Failed to process image at {local_path}: {e}")
        return ""

media_texts = []

for url in media_urls:
    print(f"Processing {url}...")
    local_path = download_image(url, image_dir)
    if not local_path:
        continue
    extracted_text = extract_text_from_image(local_path)
    if extracted_text:  
        tokenized_text = word_tokenize(extracted_text)
        media_texts.append({
            'url': url,
            'extracted_text': extracted_text,
            'tokenized_text': tokenized_text
        })

media_texts_df = pd.DataFrame(media_texts)
media_texts_df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Processing complete! Saved tokenized media texts to {output_file}. Processed {len(media_texts)} images.")
