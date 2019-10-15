### Graphviz Section ###
import pygraphviz as pgv
from deap import gp
import re

def _clean_label(label):
    label_str = str(label)

    label_str = label_str.replace("(",  "=")

    # Removing these will make interpreting the outputted tree easier
    clutter = ["Estimator", "Default", "Terminal", "Processor", "Classifier", "Type", "3", "5", ")"]

    # Remove all the words in clutter from the label
    big_regex = re.compile('|'.join(map(re.escape, clutter)))
    return big_regex.sub("", label_str)

def _graph(expr):
    nodes = list(range(len(expr)))
    edges = list()
    labels = dict()

    stack = []
    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        labels[i] = node.name if isinstance(node, gp.Primitive) else node.value
        labels[i] = _clean_label(labels[i])
        stack.append([i, node.arity])
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels


def plot_tree(tree, file_name):
    nodes, edges, labels = _graph(tree)

    # Remove the dummy nodes for visualisation
    to_remove = [idx for idx in labels if labels[idx].startswith("Dummy")]
    nodes = [node for node in nodes if node not in to_remove]

    # Need to update the edges to skip the node
    for node in to_remove:
        parents = [edge[0] for edge in edges if edge[1] == node]
        children = [edge[1] for edge in edges if edge[0] == node]

        # Remove the old edge
        edges = [edge for edge in edges if edge[0] != node and edge[1] != node]

        # Connect the children with the parents
        edges = edges + list(zip(parents, children))


    g = pgv.AGraph(outputorder="edgesfirst",  nodesep=.5, ranksep=1)
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw(file_name)
