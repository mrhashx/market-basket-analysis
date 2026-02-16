# apriori_from_scratch.py
# One-click runnable in Spyder (Run/F5) + also supports optional CLI args.
# Outputs:
#   - {out_prefix}_frequent_itemsets.csv
#   - {out_prefix}_association_rules.csv
#   - {out_prefix}_item_id_mapping.csv
#   - {out_prefix}_plots/  (PNG + PDF plots)

import argparse
import math
import itertools
from collections import defaultdict
import os

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Data prep
# -------------------------
def load_transactions(csv_path: str):
    """
    Expects at least two columns:
      - Transaction
      - Item
    Returns:
      df (raw rows),
      transactions: List[List[str]] where each inner list is unique items in a transaction
    """
    df = pd.read_csv(csv_path)
    if "Transaction" not in df.columns or "Item" not in df.columns:
        raise ValueError("CSV must contain columns: 'Transaction' and 'Item'")

    grouped = df.groupby("Transaction")["Item"].apply(lambda x: sorted(set(x)))
    transactions = grouped.tolist()
    return df, transactions


# -------------------------
# Apriori core
# -------------------------
def apriori(transactions, min_support: float):
    """
    Returns:
      freq_counts: dict[frozenset, int]   support counts
      support:     dict[frozenset, float] support fractions
    """
    N = len(transactions)
    if N == 0:
        return {}, {}

    min_count = math.ceil(min_support * N)

    # Count singletons
    item_counts = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counts[item] += 1

    Lk = {frozenset([i]): c for i, c in item_counts.items() if c >= min_count}
    freq = dict(Lk)

    k = 2
    tsets = [set(t) for t in transactions]

    while Lk:
        prev = [tuple(sorted(s)) for s in Lk.keys()]
        prev.sort()

        # Join step
        candidates = set()
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                a, b = prev[i], prev[j]
                if a[: k - 2] == b[: k - 2]:
                    cand = frozenset(set(a) | set(b))
                    if len(cand) == k:
                        candidates.add(cand)
                else:
                    break

        # Prune step
        prev_freq = set(frozenset(p) for p in prev)
        pruned = set()
        for cand in candidates:
            if all(frozenset(sub) in prev_freq for sub in itertools.combinations(cand, k - 1)):
                pruned.add(cand)

        # Count supports for candidates
        counts = defaultdict(int)
        for t in tsets:
            for cand in pruned:
                if cand.issubset(t):
                    counts[cand] += 1

        Lk = {cand: c for cand, c in counts.items() if c >= min_count}
        freq.update(Lk)
        k += 1

    support = {s: c / N for s, c in freq.items()}
    return freq, support


def generate_rules(freq_counts, support, min_conf: float):
    """
    Generates rules A -> B from frequent itemsets.
    Returns list of tuples: (A, B, support(itemset), confidence)
    """
    rules = []
    for itemset in freq_counts:
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, r):
                A = frozenset(antecedent)
                B = itemset - A
                conf = support[itemset] / support[A]
                if conf >= min_conf:
                    rules.append((A, B, support[itemset], conf))

    rules.sort(key=lambda x: (-x[3], -x[2], len(x[0]), len(x[1])))
    return rules


# -------------------------
# Plots (PNG + PDF)
# -------------------------
def save_plots(freq_df: pd.DataFrame, rules_df: pd.DataFrame, out_prefix: str,
              top_n_items: int = 20, top_n_rules: int = 20):
    plots_dir = f"{out_prefix}_plots"
    os.makedirs(plots_dir, exist_ok=True)

    def save_fig(fig, name: str):
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{name}.png"), dpi=200)
        fig.savefig(os.path.join(plots_dir, f"{name}.pdf"))
        plt.close(fig)

    # 1) Top items (k=1) by support
    one = freq_df[freq_df["k"] == 1].copy().sort_values("support", ascending=False).head(top_n_items)
    if len(one) > 0:
        fig = plt.figure()
        plt.barh(list(reversed(one["itemset"].tolist())), list(reversed(one["support"].tolist())))
        plt.xlabel("Support")
        plt.title(f"Top-{min(top_n_items, len(one))} items by Support (k=1)")
        save_fig(fig, "top_items_support")

    # 2) Support distribution of frequent itemsets
    fig = plt.figure()
    plt.hist(freq_df["support"].values, bins=20)
    plt.xlabel("Support")
    plt.ylabel("Count of frequent itemsets")
    plt.title("Support distribution of frequent itemsets")
    save_fig(fig, "freq_itemsets_support_hist")

    # 3) Top rules by confidence
    if len(rules_df) > 0:
        top_rules = rules_df.sort_values(["confidence", "support"], ascending=[False, False]).head(top_n_rules).copy()
        top_rules["rule"] = top_rules["antecedent"].astype(str) + " → " + top_rules["consequent"].astype(str)

        fig = plt.figure()
        plt.barh(list(reversed(top_rules["rule"].tolist())), list(reversed(top_rules["confidence"].tolist())))
        plt.xlabel("Confidence")
        plt.title(f"Top-{min(top_n_rules, len(top_rules))} rules by Confidence")
        save_fig(fig, "top_rules_confidence")

    # 4) Support vs Confidence scatter
    if len(rules_df) > 0:
        fig = plt.figure()
        plt.scatter(rules_df["support"].values, rules_df["confidence"].values, s=15)
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.title("Rules: Support vs Confidence")
        save_fig(fig, "rules_support_vs_confidence")

    print(f"Saved plots ✅  Folder: {plots_dir} (PNG + PDF)")


# -------------------------
# End-to-end runner
# -------------------------
def run_apriori(csv_path: str, minsup: float, minconf: float, out_prefix: str):
    df, transactions = load_transactions(csv_path)
    N = len(transactions)

    unique_items = sorted(df["Item"].unique())
    item_to_id = {item: i + 1 for i, item in enumerate(unique_items)}

    freq_counts, support = apriori(transactions, minsup)
    rules = generate_rules(freq_counts, support, minconf)

    # Frequent itemsets dataframe
    freq_rows = []
    for s, c in freq_counts.items():
        items = sorted(list(s))
        ids = [item_to_id[i] for i in items]
        freq_rows.append({
            "k": len(s),
            "itemset": ", ".join(items),
            "item_ids": ", ".join(map(str, ids)),
            "support_count": c,
            "support": round(support[s], 6),
        })
    freq_df = pd.DataFrame(freq_rows).sort_values(["k", "support_count"], ascending=[True, False])

    # Rules dataframe
    rules_rows = []
    for A, B, supp, conf in rules:
        a_items = sorted(A)
        b_items = sorted(B)
        rules_rows.append({
            "antecedent": ", ".join(a_items),
            "consequent": ", ".join(b_items),
            "support": round(supp, 6),
            "confidence": round(conf, 6),
            "antecedent_ids": ", ".join(str(item_to_id[i]) for i in a_items),
            "consequent_ids": ", ".join(str(item_to_id[i]) for i in b_items),
        })
    rules_df = pd.DataFrame(rules_rows).sort_values(["confidence", "support"], ascending=[False, False])

    # Mapping
    mapping_df = pd.DataFrame({"item_id": list(item_to_id.values()), "item": list(item_to_id.keys())})

    # Save outputs
    freq_df.to_csv(f"{out_prefix}_frequent_itemsets.csv", index=False)
    rules_df.to_csv(f"{out_prefix}_association_rules.csv", index=False)
    mapping_df.to_csv(f"{out_prefix}_item_id_mapping.csv", index=False)

    # Save plots
    save_plots(freq_df, rules_df, out_prefix=out_prefix, top_n_items=20, top_n_rules=20)

    print("========================================")
    print(f"OK ✅  Transactions: {N} | Unique items: {len(unique_items)}")
    print(f"Frequent itemsets: {len(freq_counts)} | Rules: {len(rules)}")
    print("Saved ✅")
    print(f"  - {out_prefix}_frequent_itemsets.csv")
    print(f"  - {out_prefix}_association_rules.csv")
    print(f"  - {out_prefix}_item_id_mapping.csv")
    print(f"  - {out_prefix}_plots/ (PNG + PDF)")
    print("========================================")


# -------------------------
# Main: one-click defaults + optional CLI
# -------------------------
def main():
    # Defaults for one-click Run in Spyder:
    default_csv = os.path.join(os.path.dirname(__file__), "dataset.csv")
    default_minsup = 0.01
    default_minconf = 0.30
    default_out = "run1"

    parser = argparse.ArgumentParser(description="Apriori (from scratch) for Transaction/Item dataset")
    parser.add_argument("--csv", default=default_csv, help="Path to dataset CSV (Transaction, Item)")
    parser.add_argument("--minsup", type=float, default=default_minsup, help="Minimum support (e.g., 0.01)")
    parser.add_argument("--minconf", type=float, default=default_minconf, help="Minimum confidence (e.g., 0.3)")
    parser.add_argument("--out_prefix", default=default_out, help="Prefix for output CSV files")

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"CSV file not found: {args.csv}\n"
            "Put 'dataset.csv' next to this .py file (recommended),\n"
            "or run with --csv \"full/path/to/yourfile.csv\""
        )

    run_apriori(args.csv, args.minsup, args.minconf, args.out_prefix)


if __name__ == "__main__":
    main()
