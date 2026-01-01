import re

def clean_email(text):
    text = str(text).lower()
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', ' URL ', text)
    return text.strip()
