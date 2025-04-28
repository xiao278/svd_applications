import networkx
import matplotlib.cm

def get_mappings(G:networkx.Graph):
    '''returns (n2i, i2n)'''
    node_to_index = {}
    index_to_node = {}

    for i, node in enumerate(G.nodes):
        node_to_index[node] = i
        index_to_node[i] = node
    return (node_to_index, index_to_node)

def bipartite_clustered_positioning(G:networkx.Graph, INDEX:list[int]):
    top_nodes, bottom_nodes = networkx.algorithms.bipartite.sets(G)
    (node_to_index, index_to_node) = get_mappings(G)

    top_node_index = {}
    bottom_node_index = {}
    top_index_node = {}
    bottom_index_node = {}

    for node in G.nodes:
        index = node_to_index[node]
        if node in top_nodes:
            top_node_index[node] = index
            top_index_node[index] = node
        elif node in bottom_nodes:
            bottom_node_index[node] = index
            bottom_index_node[index] = node
        else:
            assert False

    top_node_order = {}
    bottom_node_order = {}

    for index in INDEX:
        node = index_to_node[index]
        if index in top_index_node:
            top_node_order[node] = len(top_node_order)
        elif index in bottom_index_node:
            bottom_node_order[node] = len(bottom_node_order)
        else:
            assert False

    x_gap = 8
    y_gap = 1

    pos = {}

    for node, order in top_node_order.items():
        pos[node] = (0, y_gap * order)

    for node, order in bottom_node_order.items():
        pos[node] = (x_gap, y_gap * order)

    return pos

def get_colormappings(G:networkx.Graph, INDEX:list[int], clusters:list[dict]):
    (node_to_index, index_to_node) = get_mappings(G)
    group_labels = {}
    for node in G.nodes:
        group_labels[node] = 0
    for index_of_INDEX in clusters[0]:
        index = INDEX[index_of_INDEX]
        node = index_to_node[index]
        group_labels[node] = 1
    unique_groups = set(group_labels.values())
    colors = matplotlib.cm.get_cmap("Pastel1", len(unique_groups))
    color_map = {group: colors(i) for i, group in enumerate(unique_groups)}
    node_colors = [color_map[group_labels[node]] for node in G.nodes]
    return node_colors