import os

CLIPS_DIR = 'C:/Users/vijay/DLProj/data/cv-corpus-22.0-delta-2025-06-20/en/clips'

files = os.listdir(CLIPS_DIR)
sample = files[:10]
print(sample)
