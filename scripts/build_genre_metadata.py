import json
import sqlite3
import pandas as pd

def build_corpus_metadata():
    """Join match_scores + MSD metadata + tagtraum genres into one CSV."""
    
    # --- Load match_scores: structure is {msd_track_id: {midi_hash: confidence}} ---
    print("Loading match_scores.json...")
    with open('data/match_scores.json') as f:
        match_scores = json.load(f)
    
    total_pairs = sum(len(midis) for midis in match_scores.values())
    print(f"  {len(match_scores)} MSD tracks, {total_pairs} total MIDI matches")
    
    # --- Load tagtraum genres ---
    print("Loading tagtraum genres...")
    genres = {}
    with open('data/msd_tagtraum_cd2.cls') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                genres[parts[0]] = parts[1]
    print(f"  {len(genres)} tracks with genre labels")
    
    # --- Load MSD metadata ---
    print("Loading MSD track metadata...")
    conn = sqlite3.connect('data/track_metadata.db')
    metadata = pd.read_sql(
        'SELECT track_id, artist_name, title, year FROM songs',
        conn
    ).set_index('track_id')
    conn.close()
    print(f"  {len(metadata)} MSD tracks with metadata")
    
    # --- Join: iterate msd_id -> midi_dict, create one row per MIDI ---
    print("Joining...")
    rows = []
    for msd_id, midi_dict in match_scores.items():
        meta = metadata.loc[msd_id] if msd_id in metadata.index else None
        genre = genres.get(msd_id)
        
        for midi_hash, conf in midi_dict.items():
            path = f'data/lmd_matched/{msd_id[2]}/{msd_id[3]}/{msd_id[4]}/{msd_id}/{midi_hash}.mid'
            rows.append({
                'midi_hash': midi_hash,
                'midi_path': path,
                'msd_track_id': msd_id,
                'match_confidence': conf,
                'artist': meta['artist_name'] if meta is not None else None,
                'title': meta['title'] if meta is not None else None,
                'year': meta['year'] if meta is not None else None,
                'genre': genre,
            })
    
    df = pd.DataFrame(rows)
    
    # --- Save and summarize ---
    df.to_csv('data/corpus_metadata.csv', index=False)
    print(f"\nWrote {len(df)} rows to data/corpus_metadata.csv")
    print(f"  Unique MIDIs: {df['midi_hash'].nunique()}")
    print(f"  Unique songs (MSD tracks): {df['msd_track_id'].nunique()}")
    print(f"  With genre: {df['genre'].notna().sum()}")
    print(f"  With year (non-zero): {(df['year'] > 0).sum()}")
    print(f"\nGenre distribution:")
    print(df['genre'].value_counts())
    print(f"\nYear distribution (decades):")
    decades = (df['year'] // 10 * 10)
    print(decades[decades > 0].value_counts().sort_index())
    
    return df

if __name__ == '__main__':
    build_corpus_metadata()