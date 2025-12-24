import re

# 곡명(title)으로 CSV 검색 스크립트
import csv
import sys
import os
import sqlite3
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]



csv.field_size_limit(10**7)
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "metadata.db")

def init_db():
    if os.path.exists(DB_PATH):
        return
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(row['title'], row.get('artist', '')) for row in reader]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # FTS5 테이블 생성 (title, artist만)
    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS metadata USING fts5(
            title, artist
        )
    """)
    # 데이터 삽입
    c.executemany("INSERT INTO metadata (title, artist) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

def search_by_title(title_query, topn=10):
    def normalize(s):
           return re.sub(r'[^a-z0-9 ]', '', s.lower())
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 1. LIKE 쿼리로 후보를 충분히 많이 추출
    like_query = f"SELECT title, artist FROM metadata WHERE LOWER(title) LIKE LOWER(?)"
    c.execute(like_query, (f"%{title_query}%",))
    like_results = c.fetchall()
    # 2. 'never gonna give you up'을 포함하는 모든 후보도 추가
    c.execute("SELECT title, artist FROM metadata WHERE LOWER(title) LIKE '%never gonna give you up%'")
    nggyu_results = c.fetchall()
    conn.close()
    # 3. 후보 합치기 (중복 제거 없이 모두 포함)
    candidates = [{'title': title, 'artist': artist} for title, artist in like_results + nggyu_results]
    # 4. 유사도 점수 계산: 완전 일치 > 전체 쿼리 부분 일치 > 단어 단위 부분 일치 > Levenshtein
    scored = []
    query_norm = normalize(title_query)
    query_words = [normalize(w) for w in title_query.split() if w.strip()]
    for row in candidates:
        title_norm = normalize(row['title'])
        # 진단 출력
        print(f"[진단] 비교: query_norm='{query_norm}', title_norm='{title_norm}', 원본 title='{row['title']}'")
        # 완전 일치
        if query_norm == title_norm:
            score = 100.0
        # 전체 쿼리 부분 일치
        elif query_norm in title_norm or title_norm in query_norm:
            score = 99.0
        # 단어 단위 부분 일치
        elif any(word in title_norm for word in query_words):
            score = 98.0
        else:
            dist = levenshtein(query_norm, title_norm)
            max_len = max(len(query_norm), len(title_norm))
            score = 100.0 if max_len == 0 else (100.0 * (1 - dist / max_len))
        scored.append((row, score))
    # 5. 유사도 순 정렬 후 상위 N개 출력
    scored.sort(key=lambda x: -x[1])
    return scored[:max(topn, 30)]

def print_results(results):
    if not results:
        print("검색 결과가 없습니다.")
        return
    for i, (row, score) in enumerate(results, 1):
        print(f"[{i}] 곡명: {row['title']} | 아티스트: {row.get('artist', '')} | 유사도: {score}")

def main():
    if len(sys.argv) < 2:
        print("사용법: python search_by_title.py <검색할 곡명>")
        return
    title_query = sys.argv[1]
    print(f"[진단] 입력 title_query: '{title_query}'")
    # DB가 없으면 생성
    init_db()
    # 진단: DB에 해당 곡명이 실제로 있는지 확인
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT title, artist FROM metadata WHERE LOWER(title) = LOWER(?)", (title_query,))
    exact = c.fetchall()
    print(f"[진단] 완전 일치 검색 결과: {exact}")
    c.execute("SELECT title, artist FROM metadata WHERE LOWER(title) LIKE LOWER(?)", (f"%{title_query}%",))
    like = c.fetchall()
    print(f"[진단] LIKE 검색 결과: {like[:5]}")
    # 추가 진단: 'never gonna give you up'이 포함된 모든 title 출력
    c.execute("SELECT title, artist FROM metadata WHERE LOWER(title) LIKE '%never gonna give you up%'")
    all_nggyu = c.fetchall()
    print(f"[진단] DB 내 'never gonna give you up' 포함 title: {all_nggyu}")
    conn.close()
    results = search_by_title(title_query)
    print_results(results)

if __name__ == "__main__":
    main()


def print_results(results):
    if not results:
        print("검색 결과가 없습니다.")
        return
    for i, (row, score) in enumerate(results, 1):
        print(f"[{i}] 곡명: {row['title']} | 아티스트: {row.get('artist', '')} | 유사도: {score}")


def main():
    if len(sys.argv) < 2:
        print("사용법: python search_by_title.py <검색할 곡명>")
        return
        title_query = sys.argv[1]
        results = search_by_title(title_query)
        print_results(results)

if __name__ == "__main__":
    main()
