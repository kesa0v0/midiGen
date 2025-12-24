import pandas as pd
df = pd.read_csv("data/processed/cleaned_metadata.csv")
print(df['genre'].value_counts())

# check_unknowns.py
import pandas as pd

# CSV 로드
df = pd.read_csv("data/processed/cleaned_metadata.csv") # 또는 작업 중인 데이터프레임

# UNKNOWN인 것들만 필터링
unknowns = df[df['genre'] == 'UNKNOWN']

# 도대체 원래 스타일이 뭐였는지 상위 20개 출력
print("=== Top 20 Unknown Styles ===")
print(unknowns['style'].value_counts().head(20))