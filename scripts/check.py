import csv
from collections import Counter


csv.field_size_limit(10**7)

csv_path = "data/raw/Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv"
counter = Counter()

with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        styles = row["music_style_scraped"]
        # 여러 스타일이 쉼표 등으로 구분되어 있다면 분리
        for style in styles.split(","):
            style = style.strip()
            if style:
                counter[style] += 1

for style, count in counter.most_common():
    print(f"{style}: {count}")