import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def analyze_data():
    df = pd.read_csv('dataset.csv')
    
    df = df[['Transaction', 'Item']].drop_duplicates()
    
    transactions = df.groupby('Transaction')['Item'].apply(set).tolist()
    total_tx = len(transactions)
    
    min_support_count = 0.01 * total_tx
    min_confidence = 0.3
    
    item_counts = {}
    for t in transactions:
        for item in t:
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1
                
    frequent_items = {}
    for item in item_counts:
        if item_counts[item] >= min_support_count:
            frequent_items[item] = item_counts[item]
            
    print("--- ۱۰ آیتم پرفروش (Single Items) ---")
    sorted_items = sorted(frequent_items.items(), key=lambda x: x[1], reverse=True)
    for i in range(min(10, len(sorted_items))):
        name, count = sorted_items[i]
        print(f"Item: {name} | Support: {count/total_tx:.4f}")

    frequent_list = list(frequent_items.keys())
    pair_counts = {}
    
    for i in range(len(frequent_list)):
        for j in range(i + 1, len(frequent_list)):
            item1 = frequent_list[i]
            item2 = frequent_list[j]
            
            count = 0
            for t in transactions:
                if item1 in t and item2 in t:
                    count += 1
            
            if count >= min_support_count:
                pair_counts[(item1, item2)] = count

    print("\n--- قوانین انجمنی (Association Rules) ---")
    print(f"{'If Buy':<15} {'Then Buy':<15} {'Support':<10} {'Confidence':<10}")
    
    rules = []
    for pair in pair_counts:
        itemA, itemB = pair
        support_pair = pair_counts[pair] / total_tx
        
        conf_A_to_B = pair_counts[pair] / frequent_items[itemA]
        if conf_A_to_B >= min_confidence:
            rules.append((itemA, itemB, support_pair, conf_A_to_B))
            
        conf_B_to_A = pair_counts[pair] / frequent_items[itemB]
        if conf_B_to_A >= min_confidence:
            rules.append((itemB, itemA, support_pair, conf_B_to_A))
            
    rules.sort(key=lambda x: x[3], reverse=True)
    for r in rules[:15]:
        print(f"{r[0]:<15} {r[1]:<15} {r[2]:<10.4f} {r[3]:<10.4f}")

analyze_data()