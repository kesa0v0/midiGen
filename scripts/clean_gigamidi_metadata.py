import pandas as pd
import numpy as np
import re
import os
import argparse
import concurrent.futures
import multiprocessing
import time
from typing import Optional, Dict, List
from pathlib import Path

# --- Configuration & Constants ---

GENRE_KEYWORDS = {
    'OST': [
        'game', 'soundtrack', 'film', 'movie', 'anime', 'theme', 'nintendo', 'psx', 
        'console', 'cinema', 'disney', 'broadway', 'musical', 'tv'
    ],
    'CLASSIC': [
        'renaissance', 'baroque', 'classical', 'romantic', 'medieval', 'choral', 'opera', 
        'modern', 'contemporary', 'gregorian', 'chamber', 'symphony', 'concerto',
        'early 20th century', 'early20thcentury', 'ancient', # 시대 구분 추가
        "waltz"
    ],
    'ROCK': [
        'metal', 'punk', 'grunge', 'hardcore', 'alternative', 'indie', 'rock', 
        'rockabilly', 'psychedelic', 'new wave'
    ],
    'ELECTRONIC': [
        'techno', 'trance', 'house', 'edm', 'dubstep', 'drum&bass', 'electronica', 
        'dance', 'disco', 'synth', 'eurodance', 'downtempo', 'breakbeat'
    ],
    'JAZZ': [
        'jazz', 'swing', 'big band', 'bop', 'fusion', 'blues', 'ragtime', 'dixieland'
    ],
    'FOLK': [
        'folk', 'country', 'traditional', 'world', 'reggae', 'latin', 'celtic', 'irish',
        'bluegrass', 'spiritual', 'gospel', 'christian', 'praise', 'worship', # 종교/민속 음악
        'italian', 'french', 'spanish', 'german', 'mexican', 'brazil', 'greek', # 국가명 (민요 등)
        'polka', 'musette', 'calypso',
        'danish', 'australian', 'dutch', 'japanese'
    ],
    'POP': [
        'pop', 'rnb', 'soul', 'funk', 'hits', 'oldies', 'karaoke', 'ballad',
        'rap', 'hip hop', 'hiphop', 'urban', # 힙합 계열 추가
        'medley', 'duet', 'instrumental', 'instrumentals', # 기타 분류 애매한 것들을 팝으로 흡수
        'party', 'wedding', 'holiday', 'christmas', 'children',
        'novelty', 'motown', 'ballroom'
    ]
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
    # 스타일이 비어있으면 'unknown_style'이라고 명시적으로 적어줌
    final_style = normalize_text(curated) if (curated and curated != 'nan') else normalize_text(scraped)
    if not final_style: 
        final_style = "unknown_style"
        
    return 'UNKNOWN', final_style

def resolve_midi_path(file_path, midi_filename, midi_root: Optional[str]) -> Optional[Path]:
    candidates = []
    if file_path is not None and not pd.isna(file_path):
        candidates.append(str(file_path))
    if midi_filename is not None and not pd.isna(midi_filename):
        candidates.append(str(midi_filename))

    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
        if midi_root:
            rooted = Path(midi_root) / candidate
            if rooted.is_file():
                return rooted
    return None

def _normalize_tempo_changes(tempo_changes):
    if not tempo_changes:
        return []
    if isinstance(tempo_changes, tuple) and len(tempo_changes) == 2:
        times, tempos = tempo_changes
        if len(times) == len(tempos):
            return [(float(t), float(tempo)) for t, tempo in zip(times, tempos)]
    changes = []
    for change in tempo_changes:
        if isinstance(change, tuple) and len(change) == 2:
            time, tempo = change
        else:
            time = getattr(change, "time", None)
            tempo = getattr(change, "tempo", None) or getattr(change, "qpm", None)
        if time is None or tempo is None:
            return []
        changes.append((float(time), float(tempo)))
    return changes

def _estimate_duration_seconds(midi_obj) -> float:
    if hasattr(midi_obj, "get_end_time"):
        try:
            return float(midi_obj.get_end_time())
        except Exception:
            pass

    max_tick = 0
    for inst in getattr(midi_obj, "instruments", []):
        for note in getattr(inst, "notes", []):
            if note.end > max_tick:
                max_tick = note.end

    if max_tick <= 0:
        return 0.0

    ticks_per_beat = getattr(midi_obj, "ticks_per_beat", None)
    if not ticks_per_beat:
        return 0.0

    tempo_changes = _normalize_tempo_changes(getattr(midi_obj, "tempo_changes", None))
    if not tempo_changes:
        tempo_changes = [(0.0, 120.0)]

    tempo_changes = sorted(tempo_changes, key=lambda item: item[0])
    if tempo_changes[0][0] > 0:
        tempo_changes.insert(0, (0.0, tempo_changes[0][1]))

    duration = 0.0
    for idx, (start_tick, tempo) in enumerate(tempo_changes):
        end_tick = max_tick
        if idx + 1 < len(tempo_changes):
            end_tick = min(max_tick, tempo_changes[idx + 1][0])
        if end_tick <= start_tick:
            continue
        seconds_per_beat = 60.0 / max(float(tempo), 1e-6)
        duration += ((end_tick - start_tick) / ticks_per_beat) * seconds_per_beat
        if end_tick >= max_tick:
            break
    return max(0.0, duration)

def calculate_quantization_score(midi_obj, grid_resolution=0.25, tolerance_ratio=0.05) -> float:
    total_notes = 0
    on_grid_notes = 0

    ticks_per_beat = getattr(midi_obj, "ticks_per_beat", None)
    if not ticks_per_beat or grid_resolution <= 0:
        return 0.0

    grid_ticks = ticks_per_beat * grid_resolution
    tolerance = grid_ticks * tolerance_ratio

    for inst in getattr(midi_obj, "instruments", []):
        if getattr(inst, "is_drum", False):
            continue
        for note in getattr(inst, "notes", []):
            total_notes += 1
            offset = note.start % grid_ticks
            distance = min(offset, grid_ticks - offset)
            if distance <= tolerance:
                on_grid_notes += 1

    if total_notes == 0:
        return 0.0
    return on_grid_notes / total_notes

def calculate_nps(midi_obj) -> float:
    total_notes = sum(len(getattr(inst, "notes", [])) for inst in getattr(midi_obj, "instruments", []))
    length_sec = _estimate_duration_seconds(midi_obj)
    if length_sec <= 0:
        return float("inf")
    return total_notes / length_sec

def is_black_midi(midi_obj, threshold_nps=50) -> bool:
    return calculate_nps(midi_obj) > threshold_nps

def _harmonic_metrics_from_grid(prog_grid: List[List[str]]):
    if not prog_grid:
        return 0, 0.0, 1.0

    unique_chords = set()
    total_steps = 0
    nc_steps = 0
    total_changes = 0

    for bar in prog_grid:
        prev_token = None
        bar_changes = 0
        for token in bar:
            if token == "-":
                token = prev_token
            if token is None:
                token = "N.C."
            total_steps += 1
            if token == "N.C.":
                nc_steps += 1
            else:
                unique_chords.add(token)
            if prev_token is None:
                prev_token = token
                continue
            if token != prev_token:
                bar_changes += 1
                prev_token = token
        total_changes += bar_changes

    avg_changes = total_changes / len(prog_grid)
    nc_ratio = (nc_steps / total_steps) if total_steps else 1.0
    return len(unique_chords), avg_changes, nc_ratio

def compute_harmonic_metrics(midi_path: Path, grid_unit: str = "1/16"):
    try:
        from note_seq import midi_io
        from src.preprocessor.chord_progression import extract_chord_grid
        from src.preprocessor.midi_metadata import DEFAULT_KEY, get_key_from_sequence, get_time_signature
    except ImportError as exc:
        raise ImportError("note_seq is required for harmonic metrics") from exc

    note_sequence = midi_io.midi_file_to_note_sequence(str(midi_path))
    ts_num, ts_den = get_time_signature(note_sequence)
    key_name = get_key_from_sequence(note_sequence) or DEFAULT_KEY
    prog_grid = extract_chord_grid(note_sequence, key_name, (ts_num, ts_den), grid_unit)
    return _harmonic_metrics_from_grid(prog_grid)

def _evaluate_midi_row(payload):
    (
        file_path,
        midi_filename,
        midi_root,
        apply_quant,
        apply_black,
        apply_harmonic,
        skip_harmonic_if_dropped,
        grid_resolution,
        quant_tolerance_ratio,
        min_quant_score,
        black_midi_nps,
        min_unique_chords,
        max_changes_per_bar,
        max_nc_ratio,
        chord_grid_unit,
    ) = payload

    quant_score = float("nan")
    nps = float("nan")
    chord_unique = float("nan")
    chord_changes = float("nan")
    nc_ratio = float("nan")

    keep = True
    missing_path = False
    load_failure = False
    dropped_quant = False
    dropped_black = False
    dropped_harmonic = False

    try:
        midi_path = resolve_midi_path(file_path, midi_filename, midi_root)
        if midi_path is None:
            missing_path = True
            return (
                keep,
                quant_score,
                nps,
                chord_unique,
                chord_changes,
                nc_ratio,
                missing_path,
                load_failure,
                dropped_quant,
                dropped_black,
                dropped_harmonic,
            )

        midi_obj = None
        if apply_quant or apply_black:
            try:
                import miditoolkit
            except ImportError:
                pass
            else:
                try:
                    midi_obj = miditoolkit.MidiFile(str(midi_path))
                except Exception:
                    load_failure = True

        if midi_obj:
            if apply_quant:
                quant_score = calculate_quantization_score(
                    midi_obj,
                    grid_resolution=grid_resolution,
                    tolerance_ratio=quant_tolerance_ratio,
                )
                if quant_score < min_quant_score:
                    keep = False
                    dropped_quant = True
            if apply_black:
                nps = calculate_nps(midi_obj)
                if nps > black_midi_nps:
                    keep = False
                    dropped_black = True

        if apply_harmonic and not (skip_harmonic_if_dropped and not keep):
            try:
                chord_unique, chord_changes, nc_ratio = compute_harmonic_metrics(
                    midi_path,
                    grid_unit=chord_grid_unit,
                )
                if chord_unique < min_unique_chords:
                    keep = False
                    dropped_harmonic = True
                if chord_changes >= max_changes_per_bar:
                    keep = False
                    dropped_harmonic = True
                if nc_ratio >= max_nc_ratio:
                    keep = False
                    dropped_harmonic = True
            except Exception:
                load_failure = True
    except Exception:
        load_failure = True

    return (
        keep,
        quant_score,
        nps,
        chord_unique,
        chord_changes,
        nc_ratio,
        missing_path,
        load_failure,
        dropped_quant,
        dropped_black,
        dropped_harmonic,
    )

def apply_midi_quality_filters(
    df: pd.DataFrame,
    midi_root: Optional[str],
    apply_quant: bool,
    apply_harmonic: bool,
    apply_black: bool,
    grid_resolution: float,
    quant_tolerance_ratio: float,
    min_quant_score: float,
    min_unique_chords: int,
    max_changes_per_bar: float,
    max_nc_ratio: float,
    black_midi_nps: float,
    chord_grid_unit: str,
    include_quality_columns: bool,
    progress_every: int = 5000,
    num_workers: int = 1,
    mp_chunksize: int = 50,
    skip_harmonic_if_dropped: bool = True,
) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    keep_mask = []

    quant_scores = []
    nps_scores = []
    unique_chords = []
    avg_changes = []
    nc_ratios = []
    kept_count = 0

    missing_paths = 0
    load_failures = 0
    dropped_quant = 0
    dropped_black = 0
    dropped_harmonic = 0

    if num_workers <= 0:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    if apply_quant or apply_black:
        try:
            import miditoolkit
        except ImportError:
            print("Warning: miditoolkit not installed, skipping quantization/black MIDI filters.")
            apply_quant = False
            apply_black = False

    note_seq_ready = apply_harmonic
    if apply_harmonic:
        try:
            import note_seq  # noqa: F401
        except ImportError:
            print("Warning: note_seq not installed, skipping harmonic filters.")
            note_seq_ready = False
            apply_harmonic = False

    total_rows = len(df)
    if progress_every <= 0:
        progress_every = 0

    def payloads():
        for row in df.itertuples(index=False):
            file_path = getattr(row, "file_path", None)
            midi_filename = getattr(row, "midi_filename", None)
            yield (
                file_path,
                midi_filename,
                midi_root,
                apply_quant,
                apply_black,
                apply_harmonic,
                skip_harmonic_if_dropped,
                grid_resolution,
                quant_tolerance_ratio,
                min_quant_score,
                black_midi_nps,
                min_unique_chords,
                max_changes_per_bar,
                max_nc_ratio,
                chord_grid_unit,
            )

    start_time = time.time()
    if num_workers > 1:
        print(f"MIDI filter: processing {total_rows} rows with {num_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_evaluate_midi_row, payloads(), chunksize=mp_chunksize)
            for idx, result in enumerate(results, start=1):
                (
                    keep,
                    quant_score,
                    nps,
                    chord_unique,
                    chord_changes,
                    nc_ratio,
                    missing_path,
                    load_failure,
                    dropped_q,
                    dropped_b,
                    dropped_h,
                ) = result

                keep_mask.append(keep)
                if keep:
                    kept_count += 1
                quant_scores.append(quant_score)
                nps_scores.append(nps)
                unique_chords.append(chord_unique)
                avg_changes.append(chord_changes)
                nc_ratios.append(nc_ratio)

                if missing_path:
                    missing_paths += 1
                if load_failure:
                    load_failures += 1
                if dropped_q:
                    dropped_quant += 1
                if dropped_b:
                    dropped_black += 1
                if dropped_h:
                    dropped_harmonic += 1

                if progress_every and idx % progress_every == 0:
                    elapsed = time.time() - start_time
                    print(f" - {idx}/{total_rows} rows processed, kept={kept_count}, elapsed={elapsed:.1f}s")
    else:
        print(f"MIDI filter: processing {total_rows} rows...")
        for idx, payload in enumerate(payloads(), start=1):
            (
                keep,
                quant_score,
                nps,
                chord_unique,
                chord_changes,
                nc_ratio,
                missing_path,
                load_failure,
                dropped_q,
                dropped_b,
                dropped_h,
            ) = _evaluate_midi_row(payload)

            keep_mask.append(keep)
            if keep:
                kept_count += 1
            quant_scores.append(quant_score)
            nps_scores.append(nps)
            unique_chords.append(chord_unique)
            avg_changes.append(chord_changes)
            nc_ratios.append(nc_ratio)

            if missing_path:
                missing_paths += 1
            if load_failure:
                load_failures += 1
            if dropped_q:
                dropped_quant += 1
            if dropped_b:
                dropped_black += 1
            if dropped_h:
                dropped_harmonic += 1

            if progress_every and idx % progress_every == 0:
                elapsed = time.time() - start_time
                print(f" - {idx}/{total_rows} rows processed, kept={kept_count}, elapsed={elapsed:.1f}s")

    if include_quality_columns:
        df["quantization_score"] = quant_scores
        df["notes_per_second"] = nps_scores
        df["unique_chords"] = unique_chords
        df["avg_chord_changes_per_bar"] = avg_changes
        df["nc_ratio"] = nc_ratios

    filtered_df = df.loc[keep_mask].reset_index(drop=True)
    print(f"MIDI filter drop count: {len(df) - len(filtered_df)}")
    if missing_paths:
        print(f" - Missing MIDI paths (kept): {missing_paths}")
    if load_failures:
        print(f" - MIDI load failures (kept): {load_failures}")
    if apply_quant:
        print(f" - Dropped by quantization score: {dropped_quant}")
    if apply_black:
        print(f" - Dropped by black MIDI (NPS): {dropped_black}")
    if apply_harmonic and note_seq_ready:
        print(f" - Dropped by harmonic filters: {dropped_harmonic}")

    return filtered_df

def main(
    input_path,
    output_path,
    midi_root=None,
    apply_quant=False,
    apply_harmonic=False,
    apply_black=False,
    grid_resolution=0.25,
    quant_tolerance_ratio=0.05,
    min_quant_score=0.3,
    min_unique_chords=3,
    max_changes_per_bar=3.0,
    max_nc_ratio=0.5,
    black_midi_nps=50.0,
    chord_grid_unit="1/16",
    include_quality_columns=False,
    progress_every=5000,
    num_workers=1,
    mp_chunksize=50,
    skip_harmonic_if_dropped=True,
):
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
    
    # # 3. audio_text_matches_score >= 0.5 (Optional but requested)
    # if 'audio_text_matches_score' in df.columns:
    #     df = df[df['audio_text_matches_score'] >= 0.5]
    
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

    # --- Step 2.5: MIDI Quality Filters ---
    if apply_quant or apply_harmonic or apply_black:
        print("Step 2.5: MIDI Quality Filters...")
        df = apply_midi_quality_filters(
            df,
            midi_root=midi_root,
            apply_quant=apply_quant,
            apply_harmonic=apply_harmonic,
            apply_black=apply_black,
            grid_resolution=grid_resolution,
            quant_tolerance_ratio=quant_tolerance_ratio,
            min_quant_score=min_quant_score,
            min_unique_chords=min_unique_chords,
            max_changes_per_bar=max_changes_per_bar,
            max_nc_ratio=max_nc_ratio,
            black_midi_nps=black_midi_nps,
            chord_grid_unit=chord_grid_unit,
            include_quality_columns=include_quality_columns,
            progress_every=progress_every,
            num_workers=num_workers,
            mp_chunksize=mp_chunksize,
            skip_harmonic_if_dropped=skip_harmonic_if_dropped,
        )

    # --- Step 3: Genre/Style Mapping ---
    print("Step 3: Genre & Style Mapping...")
    
    # Apply mapping logic
    # This returns a tuple (Genre, Style), we convert it to DataFrame columns
    genre_style = df.apply(map_genre_and_style, axis=1)
    df['genre'] = [x[0] for x in genre_style]
    df['style'] = [x[1] for x in genre_style]

    # --- Step 3.5: Drop UNKNOWN Genre ---
    # print(f"Before dropping UNKNOWN: {len(df)}")
    
    # 장르가 UNKNOWN인 행 제거
    # df = df[df['genre'] != 'UNKNOWN']
    
    # print(f"After dropping UNKNOWN: {len(df)}")

    # --- Step 4: Final Output ---
    print("Step 4: Saving final output...")
    
    required_columns = [
        'midi_filename', 'title', 'split', 'genre', 'style', 'artist', 
        'inst_type', 'structure_type', 'total_notes'
    ]
    if include_quality_columns:
        required_columns.extend([
            'quantization_score',
            'notes_per_second',
            'unique_chords',
            'avg_chord_changes_per_bar',
            'nc_ratio',
        ])
    
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
    parser.add_argument("--midi-root", type=str, default=None, help="Root directory that contains MIDI files (used for quality filters).")
    parser.add_argument("--apply-midi-filters", action="store_true", help="Enable all MIDI quality filters.")
    parser.add_argument("--filter-quantization", action="store_true", help="Enable quantization quality filter.")
    parser.add_argument("--filter-harmonic", action="store_true", help="Enable harmonic diversity filters.")
    parser.add_argument("--filter-black-midi", action="store_true", help="Enable black MIDI density filter.")
    parser.add_argument("--grid-resolution", type=float, default=0.25, help="Beat fraction for quantization grid (default 1/16 = 0.25).")
    parser.add_argument("--quant-tolerance", type=float, default=0.05, help="Tolerance ratio around the grid for quantization score.")
    parser.add_argument("--min-quant-score", type=float, default=0.3, help="Minimum quantization score to keep.")
    parser.add_argument("--min-unique-chords", type=int, default=3, help="Minimum unique chords required to keep.")
    parser.add_argument("--max-changes-per-bar", type=float, default=3.0, help="Maximum average chord changes per bar.")
    parser.add_argument("--max-nc-ratio", type=float, default=0.5, help="Maximum allowed N.C. ratio across steps.")
    parser.add_argument("--black-midi-nps", type=float, default=50.0, help="Notes-per-second threshold for black MIDI.")
    parser.add_argument("--chord-grid-unit", type=str, default="1/16", help="Grid unit for chord inference (e.g., 1/16).")
    parser.add_argument("--include-quality-columns", action="store_true", help="Include quality metrics in the output CSV.")
    parser.add_argument("--progress-every", type=int, default=5000, help="Log progress every N rows during MIDI filters (0 disables).")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker processes for MIDI filters (0=auto, 1=serial).")
    parser.add_argument("--mp-chunksize", type=int, default=50, help="Chunk size for multiprocessing map.")
    parser.add_argument("--skip-harmonic-if-dropped", action="store_true", default=True, help="Skip harmonic inference if quant/black already dropped the row.")
    parser.add_argument("--no-skip-harmonic-if-dropped", dest="skip_harmonic_if_dropped", action="store_false", help="Always run harmonic inference even if already dropped.")
    
    args = parser.parse_args()
    apply_quant = args.filter_quantization or args.apply_midi_filters
    apply_harmonic = args.filter_harmonic or args.apply_midi_filters
    apply_black = args.filter_black_midi or args.apply_midi_filters

    main(
        args.input,
        args.output,
        midi_root=args.midi_root,
        apply_quant=apply_quant,
        apply_harmonic=apply_harmonic,
        apply_black=apply_black,
        grid_resolution=args.grid_resolution,
        quant_tolerance_ratio=args.quant_tolerance,
        min_quant_score=args.min_quant_score,
        min_unique_chords=args.min_unique_chords,
        max_changes_per_bar=args.max_changes_per_bar,
        max_nc_ratio=args.max_nc_ratio,
        black_midi_nps=args.black_midi_nps,
        chord_grid_unit=args.chord_grid_unit,
        include_quality_columns=args.include_quality_columns,
        progress_every=args.progress_every,
        num_workers=args.num_workers,
        mp_chunksize=args.mp_chunksize,
        skip_harmonic_if_dropped=args.skip_harmonic_if_dropped,
    )
