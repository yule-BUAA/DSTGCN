from scipy import spatial
import pandas as pd
import networkx as nx
import random
from collections import defaultdict
from tqdm import tqdm

from utils.load_config import get_attribute
from transform_coord.coord_converter import convert_by_type

K = get_attribute("K_hop")

# 正负样本比
pos_neg_rate = 1
trainDataPercent = 0.7
validationDataPercent = 0.1

longitudeMin = 116.09608
longitudeMax = 116.71040
latitudeMin = 39.69086
latitudeMax = 40.17647

# 坐标转换
longitudeMin, latitudeMin = convert_by_type(lng=longitudeMin, lat=latitudeMin, type="g2w")
longitudeMax, latitudeMax = convert_by_type(lng=longitudeMax, lat=latitudeMax, type="g2w")

accident_path = "/home/yule/桌面/traffic_accident_data/accident.csv"
accident_data = pd.read_csv(accident_path, header=0)

edges_path = "../data/edges_data.h5"
nodes = pd.read_hdf(edges_path)

road_network_path = "../data/beijing_roadnet.gpickle"
road_network = nx.read_gpickle(road_network_path)


# 判断传入的点是否在网格中
def point_is_in_girds(longitude, latitude):
    return longitudeMin <= longitude <= longitudeMax and latitudeMin <= latitude <= latitudeMax


# 得到传入的node_list的邻居节点
def get_neighbors(network: nx.Graph, node_list):
    nodes_set = set()
    for node in node_list:
        nodes_set = nodes_set.union({n for n in network.neighbors(node)})
    return nodes_set


if __name__ == "__main__":
    all_nodes_position_list = list(zip(nodes['XCoord'].tolist(), nodes['YCoord'].tolist()))
    tree_nodes = spatial.KDTree(all_nodes_position_list)

    # 事故与道路的经纬度为WGS-84,其余的均为GCJ-02,需要转换
    accident_dic = defaultdict(list)
    for index, row in tqdm(accident_data.iterrows(), total=accident_data.shape[0]):
        # 根据交通事故定位node(即道路)
        accident_longitude = row["longitude"]
        accident_latitude = row["latitude"]
        accident_time = pd.to_datetime(row["endTime"])
        # 不处理不在grids中的事故点
        if not point_is_in_girds(accident_longitude, accident_latitude):
            # print(f"filter {row}")
            continue
        # 找到距离事故点最近的node
        distance, node_id = tree_nodes.query([accident_longitude, accident_latitude], k=1)
        if distance >= 0.01 / 2:
            print(f"distance {distance}, node_id {node_id}")
            continue
        node_info = nodes.loc[node_id]
        # 不处理不在grids中的道路点
        if not point_is_in_girds(node_info['XCoord'], node_info['YCoord']):
            continue
        # 得到邻居节点,判断是否均在网格中
        near_nodes_id_set = {node_id}
        all_in = True
        for k in range(K):
            near_nodes_id_set = get_neighbors(network=road_network, node_list=near_nodes_id_set)
            for near_node_id in near_nodes_id_set:
                near_node = nodes.loc[near_node_id]
                if not point_is_in_girds(near_node['XCoord'], near_node['YCoord']):
                    all_in = False
                    break
            if not all_in:
                break
        # 若有邻居节点不在网格中,则忽略该点
        if not all_in:
            continue
        print(near_nodes_id_set)
        # 判断时间,应该有前24小时的速度
        if accident_time < pd.to_datetime("2018-08-02"):
            continue
        accident_dic["longitude"].append(node_info['XCoord'])
        accident_dic["latitude"].append(node_info['YCoord'])
        accident_dic["time"].append(accident_time)
        accident_dic["node_id"].append(node_id)
        accident_dic["accident"].append(1)
        # print(f"append node_id {node_id}")
        # if len(accident_dic["accident"]) >= 10:
        #     break

    # 正样本
    select_accident = pd.DataFrame(accident_dic)
    print(select_accident)
    # 负样本
    not_accident_dict = defaultdict(list)
    count = 0
    while count < len(select_accident) // pos_neg_rate:
        # 随机筛选道路点作为交通事故发生点,随机生成时间
        index_id = random.randint(0, len(nodes.index)-1)
        random_node_id = nodes.index[index_id]
        random_node = nodes.loc[random_node_id]
        date_range = pd.date_range(start="2018-08-02", end="2018-11-01", freq="1H")[:-1]
        date_index = random.randint(0, len(date_range)-1)
        random_time = date_range[date_index]
        # 不处理不在grids中的node
        if not point_is_in_girds(random_node["XCoord"], random_node["YCoord"]):
            continue
        # 判断该点没发生事故
        duplicate = select_accident[select_accident.apply(lambda x: x['longitude'] == random_node['XCoord'] and
                                    x['latitude'] == random_node['YCoord'] and
                                    pd.to_datetime(x['time']).strftime("%Y%m%d %H") == random_time.strftime("%Y%m%d %H"), axis=1)]
        if len(duplicate) > 0:
            print("duplicate")
            continue
        # 得到邻居节点,判断是否均在网格中
        near_nodes_id_set = {random_node_id}
        all_in = True
        for k in range(K):
            near_nodes_id_set = get_neighbors(network=road_network, node_list=near_nodes_id_set)
            for near_node_id in near_nodes_id_set:
                near_node = nodes.loc[near_node_id]
                if not point_is_in_girds(near_node['XCoord'], near_node['YCoord']):
                    all_in = False
                    break
            if not all_in:
                break

        # 若有邻居节点不在网格中,则忽略该点
        if not all_in:
            continue
        not_accident_dict["longitude"].append(random_node["XCoord"])
        not_accident_dict["latitude"].append(random_node["YCoord"])
        not_accident_dict["time"].append(random_time)
        not_accident_dict["node_id"].append(random_node_id)
        not_accident_dict["accident"].append(0)
        count += 1
        print(f"generate {count} negative samples")

    select_not_accident = pd.DataFrame(not_accident_dict)
    print(select_not_accident)
    # shuffle
    select_accident = select_accident.sample(frac=1).reset_index(drop=True)
    select_not_accident = select_not_accident.sample(frac=1).reset_index(drop=True)

    dataH5 = pd.HDFStore(f"../data/accident_{K}.h5", 'w')

    pos_train_idx, pos_validate_idx = int(trainDataPercent * len(select_accident)), int(
        (trainDataPercent + validationDataPercent) * len(select_accident))

    neg_train_idx, neg_validate_idx = int(trainDataPercent * len(select_not_accident)), int(
        (trainDataPercent + validationDataPercent) * len(select_not_accident))

    dataH5["train"] = pd.concat([select_accident.iloc[:pos_train_idx], select_not_accident.iloc[:neg_train_idx]], axis=0, ignore_index=True)
    dataH5["validate"] = pd.concat([select_accident.iloc[pos_train_idx:pos_validate_idx],
                                    select_not_accident.iloc[neg_train_idx:neg_validate_idx]], axis=0, ignore_index=True)
    dataH5["test"] = pd.concat([select_accident.iloc[pos_validate_idx:], select_not_accident.iloc[neg_validate_idx:]], axis=0, ignore_index=True)
    dataH5.close()
