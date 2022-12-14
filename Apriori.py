import clementine as ct
import numpy as np
from efficient_apriori import apriori
transactions = [('eggs', 'bacon', 'soup'),
                ('eggs', 'bacon', 'apple'),
                ('soup', 'bacon', 'banana')]
# transactions = ct.load_Transactions()
# itemsets, rules = apriori(transactions, min_support=0.5, min_confidence=1)
# print(rules)  # [{eggs} -> {bacon}, {soup} -> {bacon}]
# print(itemsets)
if __name__ == '__main__':
    transactions = ct.load_Transactions()
    # transactions = [('eggs', 'bacon', 'soup'),
    #                 ('eggs', 'bacon', 'apple'),
    #                 ('soup', 'bacon', 'banana')]
    trace= []
    begin = 0.05
    step= 0.05
    for i in np.arange(begin, 1+begin, begin):
        for j in np.arange(begin, 1+begin, begin):

            item_sets, rules = apriori(transactions, min_support=round(i,2), min_confidence=round(j,2))
            if len(rules) != 0 :
                print('min_support:%.2f'%i,'min_confidence:%.2f  -->'%j)
                print(rules)
                print("")
                trace.append(rules)
            # print('min_support:%2f'%i,'min_confidence:%2f  -->'%j)

