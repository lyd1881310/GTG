# from graph_tool.all import *
from graph_tool.all import Graph, shortest_distance


def build_graph(network, costs):
    g = Graph(directed=True)
    wt = g.new_edge_property('double')
    edge_list = []
    for edge in network.edge_index.t().tolist():
        edge_list.append((edge[0], edge[1], (costs[edge[0]].item() + costs[edge[1]].item()) / 2))
    g.add_edge_list(edge_list, eprops=[wt])
    g.ep.wt = wt
    return g


def shortest_path(graph, source, target):
    """
    计算最短路段序列
    :param graph: 图
    :param source:
    :param target:
    :return: list of node index
    """
    dist, pred = shortest_distance(graph, source=source, target=target, pred_map=True, weights=graph.ep.wt)
    path = [target]
    while path[-1] != source:
        path.append(pred[path[-1]])
    return path[::-1]
