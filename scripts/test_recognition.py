"""Smoke test for the recognition pipeline.

For each sampled song from the index:
  1. Slice out a contiguous chunk of its fingerprint to simulate a user query.
  2. Run recognize() and check whether the original song is the top match.
  3. Collect diagnostics on failures to understand where the system breaks.

Reports top-1 and top-5 accuracy at the song level (artist + title).
"""

import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recognition import (
    load_index, build_ngram_index, recognize, find_true_hash_rank,
    NGRAM_SIZE,
)


N_QUERIES = 50
QUERY_LEN = 80
RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)
    
    print("Loading index...")
    index = load_index()
    fingerprints = index["fingerprints"]
    metadata = index["metadata"]
    print(f"  {len(fingerprints)} fingerprints in index")
    
    by_song = defaultdict(list)
    for h, m in metadata.items():
        key = (m.get("artist", "?"), m.get("title", "?"))
        by_song[key].append(h)
    multi = sum(1 for v in by_song.values() if len(v) > 1)
    print(f"  {len(by_song):,} unique songs, {multi:,} with multiple transcriptions")
    
    # Build a hash -> set-of-sibling-hashes map for sibling-rank diagnostics
    hash_to_siblings = {}
    for hashes in by_song.values():
        s = set(hashes)
        for h in hashes:
            hash_to_siblings[h] = s
    
    print(f"Building n-gram index (n={NGRAM_SIZE})...")
    t0 = time.time()
    ngram_index, idf = build_ngram_index(fingerprints)
    print(f"  built in {time.time() - t0:.1f}s, {len(ngram_index):,} unique n-grams")
    
    eligible = [h for h, fp in fingerprints.items() if len(fp) >= QUERY_LEN + 20]
    sample_hashes = random.sample(eligible, N_QUERIES)
    
    top1 = 0
    top5 = 0
    total_time = 0.0
    
    # Failure categorization
    fail_categories = defaultdict(int)
    failure_details = []
    
    for i, true_hash in enumerate(sample_hashes, 1):
        full_fp = fingerprints[true_hash]
        start = random.randint(0, len(full_fp) - QUERY_LEN)
        query = full_fp[start:start + QUERY_LEN]
        
        t0 = time.time()
        results, diag = recognize(
            query, index, ngram_index=ngram_index, idf=idf,
            return_diagnostics=True,
        )
        total_time += time.time() - t0
        
        truth_meta = metadata.get(true_hash, {})
        truth_song = (truth_meta.get("artist"), truth_meta.get("title"))
        
        predicted_songs = [
            (r["metadata"].get("artist"), r["metadata"].get("title"))
            for r in results
        ]
        is_top1 = bool(predicted_songs) and predicted_songs[0] == truth_song
        is_top5 = truth_song in predicted_songs
        top1 += is_top1
        top5 += is_top5
        
        truth_label = f"{truth_song[0]} - {truth_song[1]}"
        status = "✓" if is_top1 else ("~" if is_top5 else "✗")
        print(f"  [{i:2d}/{N_QUERIES}] {status}  {truth_label}")
        
        if not is_top5:
            # Sibling-aware prefilter rank: was *any* sibling transcription
            # of the true song retrieved by the prefilter?
            siblings = hash_to_siblings.get(true_hash, {true_hash})
            best_sibling_rank = None
            best_sibling_score = 0.0
            for sib in siblings:
                rank, score = find_true_hash_rank(query, ngram_index, idf, sib)
                if rank is not None and (best_sibling_rank is None or rank < best_sibling_rank):
                    best_sibling_rank = rank
                    best_sibling_score = score
            
            # Categorize the failure
            if best_sibling_rank is None:
                category = "prefilter_zero_signal"
            elif best_sibling_rank > 100:
                category = "prefilter_buried"
            else:
                category = "dtw_misrank"
            fail_categories[category] += 1
            
            failure_details.append({
                "query_idx": i,
                "truth": truth_label,
                "category": category,
                "rarest_df": diag["rarest_query_ngram_df"],
                "median_df": diag["median_query_ngram_df"],
                "ngrams_in_corpus": diag["n_query_ngrams_in_corpus"],
                "sibling_rank": best_sibling_rank,
            })
            
            if results:
                pred = predicted_songs[0]
                print(f"           predicted:    {pred[0]} - {pred[1]}")
            print(f"           category:     {category}")
            print(f"           sibling rank: {best_sibling_rank}")
            print(f"           query ngrams in corpus: {diag['n_query_ngrams_in_corpus']}/{diag['n_query_ngrams']}, "
                  f"rarest df: {diag['rarest_query_ngram_df']}, median df: {diag['median_query_ngram_df']}")
    
    print()
    print(f"Top-1 accuracy (song-level): {top1}/{N_QUERIES} = {top1/N_QUERIES:.1%}")
    print(f"Top-5 accuracy (song-level): {top5}/{N_QUERIES} = {top5/N_QUERIES:.1%}")
    print(f"Avg query time: {total_time/N_QUERIES*1000:.0f} ms")
    print()
    print("Failure categories:")
    for cat, n in sorted(fail_categories.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()