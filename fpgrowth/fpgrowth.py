#!/usr/bin/env python

import itertools
from fptree import FPTree

def load_transactions(in_file):
    import csv
    handle = open(in_file,"rb") 
    reader = csv.reader(handle)
    index = 0
    transactions = []
    header = None
    for line in reader:
        index = index + 1
        if index == 1:
            header = line
        content = []
        for pos,item in enumerate(line):
            if item == "1":
                content.append(header[pos])
        transactions.append(content)
    return transactions

def generate_association_rules(patterns, confidence_threshold):
    rules = {}
    for itemset in patterns.keys():
        upper_support = patterns[itemset]

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support

                    if confidence >= confidence_threshold:
                        rules[antecedent] = (consequent, confidence)

    return rules

if __name__ == "__main__":
    big_dataset = load_transactions("data/Transactions.csv")

    support_threshold = int(len(big_dataset)*0.05)
    tree = FPTree(big_dataset, support_threshold, None, None)
    patterns = tree.mine_patterns(support_threshold)

    print "Frequent patterns:", patterns
    print "Patterns found:", len(patterns)

    # Generate association rules from the frequent itemsets.
    min_confidence = 0.5
    rules = generate_association_rules(patterns, min_confidence)
    for rule in rules.keys():
        print rule, "=>", rules[rule]
    print "Number of rules found:", len(rules)

