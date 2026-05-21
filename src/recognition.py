"""Song recognition: search the fingerprint index for the best match to a query."""

from collections import Counter, defaultdict
import math
import pickle

import numpy as np


# ---------- Tunable parameters (defaults) ----------

NGRAM_SIZE = 8       # length of interval n-grams for fast prefilter
TOP_K = 100          # number of candidates passed from prefilter to DTW
DTW_BAND = 0.1       # Sakoe-Chiba band as a fraction of sequence length (0 = no constraint)


# ---------- Index loading ----------

def load_index(path="data/fingerprint_index.pkl"):
    """Load the fingerprint index from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- Stage 1: n-gram prefilter with TF-IDF ----------

def fingerprint_to_ngrams(fingerprint, n=NGRAM_SIZE):
    """Slide a window of size n over the fingerprint; yield each window as a tuple."""
    if len(fingerprint) < n:
        return []
    return [tuple(fingerprint[i:i + n]) for i in range(len(fingerprint) - n + 1)]


def build_ngram_index(fingerprints, n=NGRAM_SIZE):
    """Build an inverted index plus IDF weights.
    
    Returns:
        ngram_index: dict mapping ngram -> set of song hashes containing it
        idf: dict mapping ngram -> log(N / df) where df = number of songs containing it.
             Rare n-grams get high weight; common ones get low weight.
    """
    ngram_index = defaultdict(set)
    for song_hash, fp in fingerprints.items():
        for ng in fingerprint_to_ngrams(fp, n):
            ngram_index[ng].add(song_hash)
    
    n_songs = len(fingerprints)
    idf = {ng: math.log(n_songs / len(hashes)) for ng, hashes in ngram_index.items()}
    
    return ngram_index, idf


def prefilter(query_fp, ngram_index, idf, top_k=TOP_K, n=NGRAM_SIZE):
    """Return the top_k candidate song hashes by TF-IDF-weighted n-gram votes.
    
    Each query n-gram contributes its IDF score to every candidate that contains it.
    A query n-gram present in many songs contributes little; a rare n-gram contributes a lot.
    """
    query_ngrams = fingerprint_to_ngrams(query_fp, n)
    if not query_ngrams:
        return []
    
    score = defaultdict(float)
    for ng in query_ngrams:
        weight = idf.get(ng)
        if weight is None:
            continue  # n-gram never seen in corpus → carries no signal
        for song_hash in ngram_index[ng]:
            score[song_hash] += weight
    
    # Return as list of (hash, score) sorted descending
    return sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


# ---------- Stage 2: DTW alignment ----------

def dtw_distance(query, reference, band=DTW_BAND):
    """Subsequence DTW: find the best contiguous alignment of `query` within `reference`.
    
    The query must align in full, but it may start and end anywhere in the reference.
    This is the right shape for matching a short user excerpt against a full-song
    fingerprint. Returns the per-element cost of the best subsequence alignment.
    
    Uses a Sakoe-Chiba band relative to the query length to constrain warping.
    """
    n, m = len(query), len(reference)
    if n == 0 or m == 0:
        return float("inf")
    
    w = max(int(band * n), 1) if band > 0 else max(n, m)
    
    # Standard DTW matrix, but row 0 of the reference dimension is all 0 —
    # this lets the alignment start anywhere in the reference for free.
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, :] = 0.0  # free start anywhere in reference
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(query[i - 1] - reference[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1],
            )
    
    # Free end: take the minimum over the last row (query fully consumed,
    # reference can end anywhere). Normalize by query length only.
    return dtw[n, :].min() / n


# ---------- The full recognizer ----------

def recognize(
    query_fp,
    index,
    ngram_index=None,
    idf=None,
    n=NGRAM_SIZE,
    top_k=TOP_K,
    band=DTW_BAND,
    return_top=5,
    return_diagnostics=False,
):
    """Identify the song that best matches a query fingerprint.
    
    Args:
        query_fp: list of intervals (the user's playing, as a fingerprint).
        index: the loaded fingerprint index (dict with 'fingerprints' and 'metadata').
        ngram_index, idf: precomputed inverted index + IDF weights.
                          If either is None, both are rebuilt on the fly.
        n, top_k, band: tunable matching parameters.
        return_top: how many ranked matches to return.
        return_diagnostics: if True, return (results, diagnostics) where diagnostics
            is a dict explaining what the prefilter saw.
    
    Returns:
        List of dicts sorted best-to-worst:
            [{'hash', 'metadata', 'dtw_distance', 'prefilter_score'}]
        If return_diagnostics=True, returns (results, diagnostics_dict).
    """
    fingerprints = index["fingerprints"]
    metadata = index["metadata"]
    
    if ngram_index is None or idf is None:
        ngram_index, idf = build_ngram_index(fingerprints, n)
    
    candidates = prefilter(query_fp, ngram_index, idf, top_k, n)
    if not candidates:
        results = []
        if return_diagnostics:
            return results, _empty_diagnostics(query_fp, ngram_index, idf, n)
        return results
    
    scored = []
    for song_hash, score in candidates:
        dist = dtw_distance(query_fp, fingerprints[song_hash], band)
        scored.append({
            "hash": song_hash,
            "metadata": metadata.get(song_hash, {}),
            "dtw_distance": dist,
            "prefilter_score": score,
        })
    
    scored.sort(key=lambda x: x["dtw_distance"])
    results = scored[:return_top]
    
    if return_diagnostics:
        diag = _build_diagnostics(query_fp, candidates, ngram_index, idf, n)
        return results, diag
    return results


# ---------- Diagnostics ----------

def _empty_diagnostics(query_fp, ngram_index, idf, n):
    return {
        "n_query_ngrams": max(0, len(query_fp) - n + 1),
        "n_query_ngrams_in_corpus": 0,
        "rarest_query_ngram_df": None,
        "median_query_ngram_df": None,
        "prefilter_candidates": 0,
        "true_hash_in_topK": None,
        "true_hash_prefilter_rank": None,
    }


def _build_diagnostics(query_fp, candidates, ngram_index, idf, n):
    """Summarize what the prefilter actually saw for this query."""
    query_ngrams = fingerprint_to_ngrams(query_fp, n)
    
    # Document frequencies (how many songs contain each query n-gram)
    dfs = [len(ngram_index[ng]) for ng in query_ngrams if ng in ngram_index]
    
    return {
        "n_query_ngrams": len(query_ngrams),
        "n_query_ngrams_in_corpus": len(dfs),
        "rarest_query_ngram_df": min(dfs) if dfs else None,
        "median_query_ngram_df": int(np.median(dfs)) if dfs else None,
        "prefilter_candidates": len(candidates),
        # These get filled in by the caller (test script) since recognize()
        # doesn't know which song is "the right answer."
        "true_hash_in_topK": None,
        "true_hash_prefilter_rank": None,
    }


def find_true_hash_rank(query_fp, ngram_index, idf, true_hash, n=NGRAM_SIZE):
    """Given a query and the known-correct hash, find where it ranked in the
    prefilter's full output (not just top-K). Returns (rank, score) or (None, 0)
    if the true hash got zero votes."""
    query_ngrams = fingerprint_to_ngrams(query_fp, n)
    score = defaultdict(float)
    for ng in query_ngrams:
        weight = idf.get(ng)
        if weight is None:
            continue
        for song_hash in ngram_index[ng]:
            score[song_hash] += weight
    
    if true_hash not in score:
        return None, 0.0
    
    # Rank = number of songs with strictly higher score, + 1
    true_score = score[true_hash]
    rank = 1 + sum(1 for s in score.values() if s > true_score)
    return rank, true_score