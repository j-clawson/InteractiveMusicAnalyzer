import pandas as pd

def clean_metadata():
    """Apply genre merges, decade bucketing, and add a 'trusted' filter."""
    
    df = pd.read_csv('../data/corpus_metadata.csv')
    print(f"Starting with {len(df)} rows")
    
    # --- Merge sparse genres into broader categories ---
    genre_map = {
        'Rock': 'Rock',
        'Punk': 'Rock',         # too sparse standalone
        'Metal': 'Rock',        # closely related, small standalone
        'Pop': 'Pop',
        'Country': 'Country',
        'Electronic': 'Electronic',
        'RnB': 'RnB',
        'Rap': 'RnB',           # MIR convention groups these
        'Jazz': 'Jazz',
        'Blues': 'Jazz',        # related, small standalone
        'Latin': 'World',
        'World': 'World',
        'Reggae': 'World',
        'Folk': 'Folk',
        'New Age': 'Other',
    }
    df['genre_grouped'] = df['genre'].map(genre_map)
    
    # --- Bucket decades ---
    def bucket_decade(year):
        if pd.isna(year) or year == 0:
            return None
        if year < 1970:
            return 'pre-1970'
        elif year < 1990:
            return '1970s-80s'
        elif year < 2000:
            return '1990s'
        else:
            return '2000s+'
    df['decade_bucket'] = df['year'].apply(bucket_decade)
    
    # --- Add a 'trusted' flag for high-confidence labeled rows ---
    df['trusted'] = (
        (df['match_confidence'] >= 0.7) &
        df['genre_grouped'].notna() &
        df['decade_bucket'].notna()
    )
    
    # --- Save ---
    df.to_csv('../data/corpus_metadata.csv', index=False)
    
    # --- Summary ---
    print(f"\nTotal rows: {len(df)}")
    print(f"Trusted rows (high conf + genre + decade): {df['trusted'].sum()}")
    print(f"\nGenre groups:")
    print(df['genre_grouped'].value_counts())
    print(f"\nDecade buckets:")
    print(df['decade_bucket'].value_counts())
    print(f"\nTrusted subset — genre × decade:")
    trusted = df[df['trusted']]
    print(pd.crosstab(trusted['genre_grouped'], trusted['decade_bucket']))
    
    return df

if __name__ == '__main__':
    clean_metadata()