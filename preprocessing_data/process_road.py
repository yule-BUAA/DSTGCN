import networkx as nx
import pandas as pd

data = pd.read_csv(r'/home/huxiao/Data/Beijing_road_net/edges.csv', header=0)
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
nx.write_gpickle(g, 'data/beijing_roadnet.gpickle')

# For read road network in Beijing.
# nx.read_gpickle('data/beijing_roadnet.gpickle')


# poi_divide_num = get_attribute("poi_divide_num")
# self.poi_tree_nodes = spatial.KDTree(list(zip(self.poi['longitude'], self.poi['latitude'])))
# # 筛选 1110m / poi_divide_num 以内的poi
# _, nodes_id = self.poi_tree_nodes.query([n_lng, n_lat], k=None,
#                     distance_upper_bound=0.01 / poi_divide_num)
# selected_poi = self.poi.loc[nodes_id]
# poi_features = selected_poi.groupby('poi_type').count()['longitude'] \
#     .reindex(list(range(1, 21)), fill_value=0).to_list()
# spatial_features.append(poi_features + [node_number, road_len])
