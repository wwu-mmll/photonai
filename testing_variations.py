import itertools

#pipeline_items = {'1': 1, '2': 1, '3': 0, '4': 0, '5': 1, '6': 0, '7': 0}
pipeline_items = [1, 1, 0, 0, 1, 0, 0]
items_to_disable = [i for i, x in enumerate(pipeline_items) if x]
print(items_to_disable)

lst = [list(i) for i in itertools.product([0, 1], repeat=len(items_to_disable))]
