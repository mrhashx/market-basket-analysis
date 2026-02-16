import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings("ignore")

def analyze_data():
    data = pd.read_csv('dataset.csv')
    
    selected_data = data[['Transaction', 'Item']]
    
    clean_data = selected_data.drop_duplicates()
    
    basket = clean_data.groupby(['Transaction', 'Item'])['Item'].count()
    basket = basket.unstack().reset_index().fillna(0)
    basket = basket.set_index('Transaction')
    
    basket_boolean = basket.astype(bool)
    
    frequent_items = apriori(basket_boolean, min_support=0.01, use_colnames=True)
    
    rules = association_rules(frequent_items, metric="confidence", min_threshold=0.3)
    
    print("--- لیست متیآهای پرفروش ---")
    frequent_items = frequent_items.sort_values(by='support', ascending=False)
    print(frequent_items.head(10))
    
    print("\n--- قوانین انجمنی استخراج شده ---")
    if not rules.empty:
        rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        rules = rules.sort_values(by='confidence', ascending=False)
        print(rules.head(15))
    else:
        print("هیچ قانونی پیدا نشد.")

analyze_data()