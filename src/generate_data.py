from pathlib import Path
import numpy as np
import pandas as pd

PROFILES = ["Gamer", "Lore Master", "Creador", "Estratega", "Tech Geek", "Explorador"]
N_QUESTIONS = 10
RNG_SEED = 42

SIGNATURES = {
    "Gamer": np.array([4.8, 1.8, 2.2, 4.4, 2.0, 2.5]),
    "Lore Master": np.array([2.1, 5.0, 3.4, 2.3, 2.2, 3.8]),
    "Creador": np.array([2.4, 3.4, 5.0, 2.6, 3.0, 3.0]),
    "Estratega": np.array([2.8, 2.5, 2.6, 5.0, 3.0, 2.9]),
    "Tech Geek": np.array([2.2, 2.4, 2.7, 3.0, 5.0, 2.8]),
    "Explorador": np.array([2.8, 3.2, 2.8, 2.6, 3.0, 5.0]),
}
QUESTION_NOISE = np.array([
    [1,.95,1,1.05,1,1],[.95,1.05,1,1,1,1],[1.05,1,.95,1,1,1],
    [1,1,1.05,.95,1,1],[1,1.05,1,1,.95,1],[1,1,1,1.05,1,.95],
    [1.05,1,1,1,1,.95],[.95,1,1,1.05,1.05,1],[1,1.05,1,.95,1.05,1],
    [1,1,1.05,1,1,.95]
])

def generate_dataset(n=3000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    labels = rng.choice(PROFILES, size=n, p=[.17,.17,.165,.165,.165,.165])
    rows=[]
    for label in labels:
        personal=rng.normal(0,.30,6)
        answers=[]
        for q in range(N_QUESTIONS):
            weights=np.clip(SIGNATURES[label]*QUESTION_NOISE[q]+personal+rng.normal(0,.55,6),.08,None)
            probs=np.exp(weights/1.25); probs/=probs.sum()
            answers.append(int(rng.choice(6,p=probs)))
        rows.append(answers+[label])
    return pd.DataFrame(rows, columns=[f"q{i}" for i in range(1,11)]+["profile"])

if __name__=="__main__":
    root=Path(__file__).resolve().parents[1]
    out=root/"data"/"geek_dataset.csv"; out.parent.mkdir(parents=True,exist_ok=True)
    df=generate_dataset(); df.to_csv(out,index=False)
    print(f"Generated {len(df):,} rows -> {out}")
    print(df["profile"].value_counts())
