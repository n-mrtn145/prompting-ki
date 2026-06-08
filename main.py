import pandas as pd

def aufbereiten():
    df = pd.read_csv('hatespeech_hocon34k.csv')

    df['text'] = df['text'].str.strip()
    df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['text'] = df['text'].str.replace('ä', 'ae', regex=False)
    df['text'] = df['text'].str.replace('ö', 'oe', regex=False)
    df['text'] = df['text'].str.replace('ü', 'ue', regex=False)
    df['text'] = df['text'].str.replace('Ä', 'Ae', regex=False)
    df['text'] = df['text'].str.replace('Ö', 'Oe', regex=False)
    df['text'] = df['text'].str.replace('Ü', 'Ue', regex=False)
    df['text'] = df['text'].str.replace('ß', 'ss', regex=False)

    df = df[df['split_12'].isin(['test', 'val'])]


    final_df = df[['post_id', 'text', 'label_hs']]

    print(f"Anzahl verarbeiteter Texte: {len(final_df)}")

    return pick50perceent(final_df)

def pick50perceent(df):
    df_hs = df[df['label_hs'] == 1]
    df_no_hs = df[df['label_hs'] == 0]

    n = min(len(df_hs), len(df_no_hs))

    df_balanced = pd.concat([
        df_hs.sample(n=n, random_state=42),
        df_no_hs.sample(n=n, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balancierter Datensatz: {n} x label_hs=1, {n} x label_hs=0 (gesamt: {len(df_balanced)})")
    return df_balanced