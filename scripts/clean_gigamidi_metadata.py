import pandas as pd
import numpy as np
import re
import os
import argparse
from typing import Optional, Dict, List

# --- Configuration & Constants ---

GENRE_KEYWORDS = {
    'OST': ['game', 'soundtrack', 'film', 'movie', 'anime', 'theme', 'nintendo', 'psx'],
    'CLASSIC': ['renaissance', 'baroque', 'classical', 'romantic', 'medieval', 'choral', 'opera', 'modern', 'contemporary'],
    'ROCK': ['metal', 'punk', 'grunge', 'hardcore', 'alternative', 'indie', 'rock'],
    'ELECTRONIC': ['techno', 'trance', 'house', 'edm', 'dubstep', 'drum&bass', 'electronica', 'dance', 'disco'],
    'JAZZ': ['jazz', 'swing', 'big band', 'bop', 'fusion', 'blues'],
    'FOLK': ['folk', 'country', 'traditional', 'world', 'reggae', 'latin', 'celtic', 'irish'],
    'POP': ['pop', 'rnb', 'soul', 'funk', 'hits', 'oldies', 'karaoke', 'ballad']
}

# OST specific keywords for the scraped data priority check
OST_SCRAPED_PRIORITY = ['game', 'nintendo', 'psx']

def normalize_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase and remove special characters (keep alphanumeric and spaces)
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def extract_relative_path(path):
    """
    Extracts relative path starting with train, validation, or test.
    """
    if pd.isna(path):
        return None
    
    # Normalize path separators
    path = path.replace('\\', '/')
    
    match = re.search(r'(train|validation|test).*', path, re.IGNORECASE)
    if match:
        return match.group(0)
    return path # Return original if pattern not found (fallback)

def determine_split(path):
    if pd.isna(path):
        return 'unknown'
    path_lower = path.lower()
    if 'train' in path_lower:
        return 'train'
    elif 'validation' in path_lower or 'valid' in path_lower:
        return 'validation'
    elif 'test' in path_lower:
        return 'test'
    return 'unknown'

def determine_inst_type(row):
    cat = row.get('instrument_category')
    tracks = row.get('num_tracks')
    
    if cat == 1:
        return 'BAND'
    elif cat == 2:
        if tracks <= 2:
            return 'PIANO_SOLO'
        else:
            return 'ORCHESTRA'
    return 'UNKNOWN' # Should be filtered out, but good for safety

def determine_structure_type(loopability):
    if pd.notna(loopability) and loopability > 0.8:
        return 'LOOP'
    return 'LINEAR'

def map_genre_and_style(row):
    curated = str(row.get('music_styles_curated', '')).lower()
    scraped = str(row.get('music_style_scraped', '')).lower()
    
    # Priority 1: Check specific OST keywords in scraped data
    for keyword in OST_SCRAPED_PRIORITY:
        if keyword in scraped:
            return 'OST', keyword
            
    # Helper to check keywords against text
    def check_keywords(text):
        for genre, keywords in GENRE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return genre, keyword
        return None, None

    # Priority 2: Check curated styles
    if curated and curated != 'nan':
        genre, keyword = check_keywords(curated)
        if genre:
            return genre, normalize_text(curated) # Use full curated string as style if matched? Or the matched keyword? Prompt says: "mapped original text... normalized"
            # We will return the normalized curated text as style.
    
    # Priority 3: Check scraped styles
    if scraped and scraped != 'nan':
        genre, keyword = check_keywords(scraped)
        if genre:
            return genre, normalize_text(scraped)

    # Priority 4: Unknown
    style = normalize_text(curated) if (curated and curated != 'nan') else normalize_text(scraped)
    return 'UNKNOWN', style

def main(input_path, output_path):
    print(f"Loading data from {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        # Create a dummy file for demonstration if it doesn't exist? 
        # No, better to fail or let the user know. 
        # But for the purpose of the CLI run, if it's missing, I can't process it.
        return

    # Load data (handling potential large file size)
    # For 1GB, reading entire file into memory is usually fine on modern systems (needs ~2-4GB RAM).
    # If OOM occurs, we would need chunking. Assuming 8GB+ RAM available.
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Rename complex columns to simple internal names
    df = df.rename(columns={
        "instrument_category: drums-only: 0, all-instruments-with-drums: 1,no-drums: 2": "instrument_category"
    })

    print(f"Original shape: {df.shape}")

    # --- Step 1: Filtering ---
    print("Step 1: Filtering data...")
    initial_count = len(df)
    
    # 1. instrument_category != 0 (Drop drums-only)
    df = df[df['instrument_category'] != 0]
    
    # 2. total_notes >= 100
    df = df[df['total_notes'] >= 100]
    
    # 3. audio_text_matches_score >= 0.5 (Optional but requested)
    if 'audio_text_matches_score' in df.columns:
        df = df[df['audio_text_matches_score'] >= 0.5]
    
    # 4. num_tracks != 0
    df = df[df['num_tracks'] != 0]
    
    print(f"Filtered {initial_count - len(df)} rows. Current shape: {df.shape}")

    # --- Step 2: Feature Engineering ---
    print("Step 2: Feature Engineering...")
    
    # midi_filename
    df['midi_filename'] = df['file_path'].apply(extract_relative_path)
    
    # split
    df['split'] = df['midi_filename'].apply(determine_split)
    
    # inst_type
    df['inst_type'] = df.apply(determine_inst_type, axis=1)
    
    # structure_type
    loop_col = 'loopability (expressive)'
    if loop_col not in df.columns:
        # Try finding alternative name or default
        loop_col = [c for c in df.columns if 'loopability' in c]
        loop_col = loop_col[0] if loop_col else None
    
    if loop_col:
        # Ensure column is numeric
        df[loop_col] = pd.to_numeric(df[loop_col], errors='coerce')
        df['structure_type'] = df[loop_col].apply(determine_structure_type)
    else:
        df['structure_type'] = 'LINEAR' # Default if column missing
        
    # artist & title
    df['artist'] = df['artist'].fillna('unknown').apply(normalize_text)
    df['title'] = df['title'].fillna('unknown').apply(normalize_text)

    # --- Step 3: Genre/Style Mapping ---
    print("Step 3: Genre & Style Mapping...")
    
    # Apply mapping logic
    # This returns a tuple (Genre, Style), we convert it to DataFrame columns
    genre_style = df.apply(map_genre_and_style, axis=1)
    df['genre'] = [x[0] for x in genre_style]
    df['style'] = [x[1] for x in genre_style]

    # --- Step 4: Final Output ---
    print("Step 4: Saving final output...")
    
    required_columns = [
        'midi_filename', 'title', 'split', 'genre', 'style', 'artist', 
        'inst_type', 'structure_type', 'total_notes'
    ]
    
    # Ensure all columns exist
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Column {col} missing, filling with defaults.")
            df[col] = 'unknown' if df[col].dtype == 'object' else 0

    final_df = df[required_columns]
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Successfully saved cleaned metadata to: {output_path}")
    print(f"Final shape: {final_df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Gigamidi Metadata CSV")
    parser.add_argument("--input", type=str, default="data/raw/Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv", help="Path to raw CSV file")
    parser.add_argument("--output", type=str, default="data/processed/cleaned_metadata.csv", help="Path to output CSV file")
    
    args = parser.parse_args()
    
    main(args.input, args.output)
