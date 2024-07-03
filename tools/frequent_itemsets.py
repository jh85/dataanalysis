from typing import List,Dict,Set,Tuple,Optional
from itertools import combinations
from collections import defaultdict
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
import time
import numpy as np
import pandas as pd
import json


#
# Apriori Algorithm
#
def generate_L1(dataset: List[List[int]],
                min_support_count: int) -> Dict[Tuple[int],Set[int]]:
    '''
    Generate L1 itemsets from given dataset
    '''
    L1 = defaultdict(set)
    for tid,transaction in enumerate(dataset):
        for item in transaction:
            L1[(item,)].add(tid)
    return {item:tids for item,tids in L1.items() if len(tids) >= min_support_count}

def has_infrequent_subset(candidate: Tuple[int,...], prev_L: Dict[Tuple[int,...],Set[int]]) -> bool:
    '''
    Check if the candidate itemset (k length) has any infrequent subset of k-1 length
    '''
    for subset in combinations(candidate, len(candidate)-1):
        if subset not in prev_L:
            return True
    return False

def apriori_gen(prev_L: Dict[Tuple[int,...], Set[int]]) -> Dict[Tuple[int,...],Set[int]]:
    '''
    Generate candidate k-itemsets from (k-1)-itemsets
    '''
    Ck = dict()
    for item1,tids1 in prev_L.items():
        for item2,tids2 in prev_L.items():
            if item1[:-1] == item2[:-1] and item1[-1] < item2[-1]:
                new_item = tuple([*item1,item2[-1]])
                if has_infrequent_subset(new_item, prev_L):
                    continue
                new_tids = set.intersection(tids1, tids2)
                Ck[new_item] = new_tids
    return Ck

def gen_Lk(Ck: Dict[Tuple[int,...], Set[int]],
           dataset: List[List[int]],
           min_support_count: int) -> Dict[Tuple[int,...],Set[int]]:
    '''
    Generate frequent k-itemsets from candidate k-itemsets
    '''
    Lk = dict()
    for candidate,tids in Ck.items():
        if len(tids) < min_support_count:
            continue
        Lk[candidate] = tids
    return Lk

def myapriori(dataset: List[List[int]],
              min_support_count: int) -> Dict[int,Dict[Tuple[int,...], Set[int]]]:
    '''
    Implement the Apriori algorithm to find frequent itemsets.

    Args
    ----
    dataset: List[List[int]]
        A list of transactions, where each transaction is a list of integers
        representing items purchased.
    min_support_count: int
        The minimum number of transactions an itemset must appear in to be considered frequent.

    Returns
    -------
    pandas.DataFrame: A DataFrame containing frequent itemsets and their support, where:
        - "support" column shows the fraction of transactions containing the itemset
        - "itemsets" column shows the frequent itemsets as tuples of items
    '''
    L1 = generate_L1(dataset, min_support_count)
    L = {1: L1}
    for k in range(2,10000):
        if len(L[k-1]) < 2:
            break
        Ck = apriori_gen(L[k-1])
        L[k] = gen_Lk(Ck, dataset, min_support_count)

    N = len(dataset)
    support = []
    itemsets = []
    for l in L.values():
        for k,v in l.items():
            support.append(len(v)/N)
            itemsets.append(k)
    df = pd.DataFrame()
    df["support"] = support
    df["itemsets"] = itemsets
    return df


#
# FP-Growth Algorithm
#
class FPNode:
    def __init__(self, value: int, count: int, parent: Optional["FPNode"]):
        self.value = value
        self.count = count
        self.parent = parent
        self.link: Optional["FPNode"] = None
        self.children: List["FPNode"] = []

    def has_child(self, value: int) -> bool:
        return any(node.value == value for node in self.children)

    def get_child(self, value: int) -> Optional["FPNode"]:
        for node in self.children:
            if node.value == value:
                return node
        return None

    def add_child(self, value: int) -> "FPNode":
        child = FPNode(value=value, count=1, parent=self)
        self.children.append(child)
        return child

class FPTree:
    def __init__(self,
                 transactions: List[List[int]],
                 min_support_count: int,
                 root_value: Optional[int],
                 root_count: Optional[int]):
        self.frequent: Dict[int, int] = self.find_frequent_items(transactions, min_support_count)
        self.headers: Dict[int, Optional["FPNode"]] = self.build_header_table(self.frequent)
        self.root: FPNode = self.build_fptree(transactions,
                                              root_value,
                                              root_count,
                                              self.frequent,
                                              self.headers)

    @staticmethod
    def find_frequent_items(transactions: List[List[int]],
                            min_support_count: int) -> Dict[int, int]:
        items: Dict[int, int] = defaultdict(int)
        for t in transactions:
            for item in t:
                items[item] += 1
        return {k:v for k,v in items.items() if v >= min_support_count}

    @staticmethod
    def build_header_table(frequent: Dict[int, int]) -> Dict[int, Optional["FPNode"]]:
        return {k:None for k in frequent.keys()}

    def build_fptree(self,
                     transactions: List[List[int]],
                     root_value: Optional[int],
                     root_count: Optional[int], 
                     frequent: Dict[int, int],
                     headers: Dict[int, Optional["FPNode"]]) -> "FPNode":
        root = FPNode(value=root_value, count=root_count, parent=None)
        for t in transactions:
            freq_items = sorted([item for item in t if item in frequent],
                                key=lambda x: frequent[x],
                                reverse=True)
            if freq_items:
                self.insert_tree(freq_items, root, headers)
        return root

    def insert_tree(self,
                    items: List[int],
                    node: "FPNode",
                    headers: Dict[int, Optional["FPNode"]]) -> None:
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.count += 1
        else:
            child = node.add_child(first)
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    current = current.link
                current.link = child

        remaining_items = items[1:]
        if remaining_items:
            self.insert_tree(remaining_items, child, headers)

    def tree_has_single_path(self, node: "FPNode") -> bool:
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 1:
            return self.tree_has_single_path(node.children[0])
        else:
            # assert num_children == 0
            return True

    def mine_patterns(self, min_support_count: int) -> Dict[Tuple[int,...], int]:
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            patterns = self.zip_patterns(self.mine_sub_trees(min_support_count))
            if self.root.value != None and tuple([self.root.value]) not in patterns:
                patterns[tuple([self.root.value])] = self.root.count
            return patterns


    def zip_patterns(self, patterns: Dict[Tuple[int,...], int]) -> Dict[Tuple[int,...], int]:
        suffix = self.root.value
        if suffix is not None:
            return {tuple(sorted(list(k) + [suffix])): v for k,v in patterns.items()}
        return patterns

    def generate_pattern_list(self) -> Dict[Tuple[int,...], int]:
        patterns: Dict[Tuple[int,...], int] = {}
        items = list(self.frequent.keys())
        if self.root.value is None:
            suffix_value: List[int] = []
        else:
            suffix_value = [self.root.value]
            patterns[tuple(suffix_value)] = self.root.count

        for i in range(1, len(items) + 1):
            for subset in combinations(items, i):
                pattern = tuple(sorted(list(subset) + suffix_value))
                patterns[pattern] = min(self.frequent[x] for x in subset)
        return patterns

    def mine_sub_trees(self, min_support_count: int) -> Dict[Tuple[int, ...], int]:
        patterns: Dict[Tuple[int, ...], int] = defaultdict(int)
        mining_order = sorted(self.frequent.keys(), key=lambda x: self.frequent[x])

        for item in mining_order:
            suffixes: List["FPNode"] = []

            # conditional_tree_input is similar to transactions.
            # It contains paths in the existing FPTree, which represent a subset
            # of the transactions that include a specific item.
            conditional_tree_input: List[List[int]] = []

            node = self.headers[item]

            while node is not None:
                suffixes.append(node)
                node = node.link

            for suffix in suffixes:
                frequency = suffix.count
                path: List[int] = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent

                # Replicate the path based on its frequency to maintain
                # accurate support counts in the conditional tree
                conditional_tree_input.extend([path] * frequency)

            subtree = FPTree(transactions=conditional_tree_input,
                             min_support_count=min_support_count,
                             root_value=item,
                             root_count=self.frequent[item])
            subtree_patterns = subtree.mine_patterns(min_support_count)

            for pattern, count in subtree_patterns.items():
                patterns[pattern] += count
        return patterns

def myfpgrowth(transactions: List[List[int]],
               min_support_count: int) -> Dict[Tuple[int,...], int]:
    """
    Executes the FP-growth algorithm to find frequent itemsets in the given transactions.
    
    Args
    ----
    transactions (List[List[int]]): A list of transactions, where each transaction is a list of items.
    min_support_count (int): The minimum number of occurrences for an itemset to be considered frequent.
    
    Returns
    -------
    pandas.DataFrame: A DataFrame containing frequent itemsets and their support, where:
        - "support" column shows the fraction of transactions containing the itemset
        - "itemsets" column shows the frequent itemsets as tuples of items
    """
    transactions_sorted = []
    for t in transactions:
        transactions_sorted.append(sorted(t))
    tree = FPTree(transactions=transactions_sorted,
                  min_support_count=min_support_count,
                  root_value=None,
                  root_count=None)
    freq_items = tree.mine_patterns(min_support_count)
    df = pd.DataFrame()
    df["support"] = np.array(list(freq_items.values())) / len(transactions)
    df["itemsets"] = list(freq_items.keys())
    return df

def generate_association_rules(patterns: Dict[Tuple[int,...], int],
                               min_confidence: float):
    rules = dict()
    for itemset in patterns.keys():
        upper_support = patterns[itemset]

        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support

                    if confidence >= min_confidence:
                        rules[antecedent] = (consequent, confidence)

    return rules



def number2name(df, id2desc): 
    new_antecedents = []
    for a in df.antecedents.tolist():
        new_antecedents.append(tuple(map(lambda x: id2desc[x], a)))
    df.drop(columns=["antecedents"])
    df["antecedents"] = new_antecedents
    
    new_consequents = []
    for a in df.consequents.tolist():
        new_consequents.append(tuple(map(lambda x: id2desc[x], a)))
    df.drop(columns=["consequents"])
    df["consequents"] = new_consequents

    return df[["antecedents","consequents","antecedent support","consequent support",
               "support","confidence","lift","leverage","conviction","zhangs_metric"]]

def main():
    pd.set_option("display.max_columns", None)
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.max_colwidth = 100
    
    filename = "online_retail_II.xlsx"
    df = pd.read_excel(filename)

    # ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
    #  'Price', 'Customer ID', 'Country']

    desc2id = dict()
    id2desc = dict()
    for id_num,desc in enumerate(set(df.Description.tolist())):
        desc2id[desc] = id_num
        id2desc[id_num] = desc

    df["Item"] = list(map(lambda d: desc2id[d], df.Description.tolist()))
    transactions = defaultdict(set)
    for i in range(len(df)):
        row = df.iloc[i,:]
        inv = row["Invoice"]
        item = int(row["Item"])
        transactions[inv] = set.union(transactions[inv], set([item]))
    dataset = [list(v) for v in transactions.values()]

    N = len(dataset)
    min_support_count = 100

    t1 = time.time()
    frequent_itemsets1 = myapriori(dataset, min_support_count)
    t2 = time.time()
    rules1 = association_rules(frequent_itemsets1, metric="confidence", min_threshold=0.93)
    rules1 = number2name(rules1, id2desc)
    
    t3 = time.time()
    frequent_itemsets2 = myfpgrowth(dataset, min_support_count)
    t4 = time.time()
    # rules2 = generate_association_rules(frequent_itemsets2, 0.93)
    rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=130)
    rules2 = number2name(rules2, id2desc)

    print(f"Apriori:   frequent itemsets = {len(frequent_itemsets1)} time = {round(t2-t1,2)} sec")
    print(f"FP-Growth: frequent itemsets = {len(frequent_itemsets2)} time = {round(t4-t3,2)} sec")
    print("Confidence > 0.93")
    print(rules1)
    print("Lift > 130")
    print(rules2)

    return
    
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    t5 = time.time()
    frequent_itemsets4 = apriori(df, min_support=min_support_count/N, use_colnames=True)
    t6 = time.time()
    rules4 = association_rules(frequent_itemsets4, metric="confidence", min_threshold=0.93)
    print(f"mlxtend apriori: frequent itemsets = {len(frequent_itemsets4)} time = {round(t6-t5,2)} sec")
    print(rules4)
    
main()

