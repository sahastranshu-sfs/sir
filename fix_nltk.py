import nltk
import os
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading NLTK data...")
resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
for resource in resources:
    try:
        print(f"Downloading {resource}...")
        nltk.download(resource)
        print(f"Successfully downloaded {resource}")
    except Exception as e:
        print(f"Failed to download {resource}: {e}")

print("NLTK data download process completed.")
print(f"NLTK data path: {nltk.data.path}")
