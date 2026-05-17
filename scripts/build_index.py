"""Build a fingerprint index from the trusted subset of corpus_metadata.csv.

For each MIDI in the trusted subset, compute its fingerprint (interval sequence)
and save the result as a pickle file containing:
  - fingerprints: dict mapping midi_hash → list of intervals
  - metadata: dict mapping midi_hash → row info (artist, title, genre, decade, etc.)
"""

import sys
import os
import pickle
import pandas as pd
from tqdm import tqdm

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.midi_utils import midi_to_fingerprint


def build_index(output_path='data/fingerprint_index.pkl', trusted_only=True):
    print("Loading corpus_metadata.csv...")
    df = pd.read_csv('data/corpus_metadata.csv')
    
    if trusted_only:
        df = df[df['trusted']]
        print(f"  Filtered to trusted subset: {len(df)} rows")
    else:
        print(f"  Using all rows: {len(df)}")
    
    fingerprints = {}
    metadata = {}
    failures = []
    
    print("Computing fingerprints...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fp = midi_to_fingerprint(row['midi_path'])
        
        if fp is None:
            failures.append({
                'midi_hash': row['midi_hash'],
                'midi_path': row['midi_path'],
                'reason': 'load_or_extract_failed'
            })
            continue
        
        fingerprints[row['midi_hash']] = fp
        metadata[row['midi_hash']] = {
            'artist': row['artist'],
            'title': row['title'],
            'genre': row['genre_grouped'],
            'decade': row['decade_bucket'],
            'year': row['year'],
            'msd_track_id': row['msd_track_id'],
            'match_confidence': row['match_confidence'],
        }
    
    print(f"\nSuccessfully fingerprinted: {len(fingerprints)}")
    print(f"Failures: {len(failures)}")
    
    # Save
    output = {
        'fingerprints': fingerprints,
        'metadata': metadata,
        'failures': failures,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nWrote index to {output_path}")
    
    # Quick stats on fingerprint lengths
    lengths = [len(fp) for fp in fingerprints.values()]
    if lengths:
        print(f"\nFingerprint length stats:")
        print(f"  min:    {min(lengths)}")
        print(f"  median: {sorted(lengths)[len(lengths)//2]}")
        print(f"  max:    {max(lengths)}")
        print(f"  mean:   {sum(lengths)/len(lengths):.0f}")


if __name__ == '__main__':
    build_index()
