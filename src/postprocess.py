import pandas as pd


def location_voting(df, threshold=0.7):
    df_pot = df[df["damage_type"] == "p"]

    results = []

    for loc, g in df_pot.groupby("location_id"):
        correct = ((g["true_class"] == 1) & (g["pred_class"] == 1)).sum()
        frac = correct / len(g)

        if frac >= threshold:
            results.append(loc)

    return results