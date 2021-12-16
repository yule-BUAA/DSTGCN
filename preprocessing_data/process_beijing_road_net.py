import networkx as nx
import pandas as pd

data = pd.read_csv(r'../original_data/Beijing_road_net_data/Beijing_Edgelist.csv', header=0)
edges_as_nodes = data.groupby('EDGE').agg({'XCoord': 'mean',
                                           'YCoord': 'mean',
                                           'START_NODE': 'nunique',
                                           'END_NODE': 'nunique',
                                           'LENGTH': 'mean'})
edges_as_nodes['NUM_NODE'] = edges_as_nodes['START_NODE']
edges_as_nodes.drop(['START_NODE', 'END_NODE'], axis=1, inplace=True)

g = nx.DiGraph()
g.add_nodes_from(edges_as_nodes.to_dict('index').items())

adjacency_as_edges = set()
edges = data.drop(['XCoord', 'YCoord', 'LENGTH'], axis=1)
adjacency = pd.merge(edges, edges, left_on='START_NODE', right_on='END_NODE')[['EDGE_x', 'EDGE_y']]
adjacency = adjacency[adjacency['EDGE_x'] != adjacency['EDGE_y']]
adjacency_as_edges = adjacency_as_edges.union(
    set(map(lambda record: (record['EDGE_x'], record['EDGE_y']), adjacency.to_dict('records'))))

adjacency = pd.merge(edges, edges, left_on='END_NODE', right_on='START_NODE')[['EDGE_x', 'EDGE_y']]
adjacency = adjacency[adjacency['EDGE_x'] != adjacency['EDGE_y']]
adjacency_as_edges = adjacency_as_edges.union(
    set(map(lambda record: (record['EDGE_x'], record['EDGE_y']), adjacency.to_dict('records'))))

g.add_edges_from(adjacency_as_edges)
g = nx.convert_node_labels_to_integers(g)
nx.write_gpickle(g, '../data/beijing_roadnet.gpickle')
print("beijing_roadnet.gpickle writes successfully.")
