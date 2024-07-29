import numpy as np
import csv
import math

def read_data(filename):
    with open(filename, 'r') as f:
        datareader = csv.reader(f)
        metadata = next(datareader)
        return metadata, list(datareader)

class Node:
    def _init_(self, attribute="", answer=""):
        self.attribute = attribute
        self.children = []
        self.answer = answer

def subtables(data, col, delete):
    items = np.unique(data[:, col])
    dict = {item: data[data[:, col] == item] for item in items}
    if delete:
        dict = {k: np.delete(v, col, 1) for k, v in dict.items()}
    return items, dict

def entropy(S):
    _, counts = np.unique(S, return_counts=True)
    probs = counts / S.size
    return -np.sum(probs * np.log2(probs))

def gain_ratio(data, col):
    total_entropy = entropy(data[:, -1])
    items, dict = subtables(data, col, delete=False)
    total_size = data.shape[0]
    entropies = sum((dict[item].shape[0] / total_size) * entropy(dict[item][:, -1]) for item in items)
    iv = sum(-((dict[item].shape[0] / total_size) * np.log2(dict[item].shape[0] / total_size)) for item in items)
    return (total_entropy - entropies) / iv

def create_node(data, metadata):
    if np.unique(data[:, -1]).size == 1:
        return Node(answer=data[0, -1])
    gains = [gain_ratio(data, col) for col in range(data.shape[1] - 1)]
    split = np.argmax(gains)
    node = Node(attribute=metadata[split])
    items, dict = subtables(data, split, delete=True)
    node.children = [(item, create_node(subtable, np.delete(metadata, split))) for item, subtable in dict.items()]
    return node

def print_tree(node, level=0):
    indent = " " * level * 2
    if node.answer:
        print(f"{indent}{node.answer}")
    else:
        print(f"{indent}{node.attribute}")
        for value, child in node.children:
            print(f"{indent}  {value}")
            print_tree(child, level + 2)

metadata, traindata = read_data("tennisdata.csv")
data = np.array(traindata)
node = create_node(data, metadata)
print_tree(node)
