# Deep Spatio-Temporal Graph Convolutional Network for Traffic Accident Prediction (DSTGCN)

DSTGCN is a graph-based neural network that could predict the risk of traffic accidents in the future.

Please refer to our Neurocomputing 2020 paper [“**Deep Spatio-Temporal Graph Convolutional Network for Traffic Accident Prediction**”]() for more details.

## Project Structure
The descriptions of principal files in this project are introduced as follows:
- ./model/
    - `spatial_layer.py`: codes for the Spatial Convolutional Layer
    - `st_gcn.py`: codes for the Spatial-Temporal Convolutional Block
    - `fully_connected.py`: codes for the fully connected layer (used for reducing the dimension of external factor and making prediction)
- ./preprocessing_data/
  - `process_xxx.py`: process the raw 'xxx' dataset
  - `generate_data.py`: codes for generating the dataset for our model
- ./transform_coord/: including files to convert the coordinate, which is download [here](https://github.com/wandergis/coordTransform_py).
- ./train/
  - `train_model.py` and `train_main.py`: codes for training models
- ./test/
  - `testing_model.py`: codes for evaluating models
- ./utils/: containing useful files that are required in the project (e.g. data loader, metrics calculation, loss function, configurations) 
- ./data/: processed datasets are under in this folder. Due to the privacy issue datasets, you could use the preprocess codes in ./preprocessing_data/ folder to generate your own datasets and use them to train the model. 
- ./saves/ and ./runs/: folders to save models and outputs of tensorboardX respectively
- ./results/: folders to save the evaluation metrics for models. 

## Formats and descriptions of the processed datasets:
accident data format: records of traffic accidents

| "longitude" | "latitude" | "startTime" | "endTime" |
|  ----  | ----  | ----  | ----  | 
| accident longitude | accident latitude | accident start time | accident end time | 

poi data format: records of pois

| "longitude" | "latitude" | "poi_type" |
| ----  | ----  | ----  | 
| poi longitude | poi latitude | poi function type | 

road data (beijing_roadnet.gpickle) format: records of road network

type: networkx.classes.graph.Graph, road network structure that records the connectivity of road segments  

speed data (all_grids_speed) format: records of speed in each grid

type: pandas.core.frame.DataFrame, containing the traffic speed of each grid

weather data format: records of the weather condition

| "temp" | "dewPt" | "pressure" | "wspd" | ... |
| ----  | ----  | ----  |  ----  | ----  | 
| temperature | dew point | pressure | wind speed | etc. | 

edge.h5 data format: records of spatial features (poi and road segment features)

| "XCoord" | "YCoord" | "LENGTH" | "NUM_NODE" | "spatial_features" |
| ----  | ----  | ----  |  ----  | ----  | 
| road segment longitude | road segment latitude | road segment length | points that road segment contains | road segment poi distribution (a list of each poi type numbers) | 

## Parameter Settings
Please refer to our paper for more details of parameter settings. 
Hyperparameters could be found in ./utils/config.json and you can adjust them when running the model.

## How to use
- Training: after setting the parameters, run ```python train_main.py``` to train models. 
- Testing: figure out the path of the specific saved model (i.e. variable ```model_path``` in ./test/testing_model.py) and then run ```python testing_model.py``` to evaluate models.

Principal environmental dependencies as follows:
- [PyTorch 1.5.0](https://pytorch.org/)
- [dgl 0.4.3](https://www.dgl.ai/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [sklearn](https://scikit-learn.org/stable/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)

## Citation
Please consider citing the following paper when using our code.

```bibtex
@article{DSTGCN,
  title     = {Deep Spatio-Temporal Graph Convolutional Network for Traffic Accident Prediction},
  author    = {Le Yu, Bowen Du, Xiao Hu, Leilei Sun, Liangzhe Han and Weifeng Lv},
  journal   = {Neurocomputing},
  year      = {2020}
}
```


特征包括:路网(长度),时间特征(速度,天气),空间特征(POI),外部特征(calendar)

经纬度统一表示的坐标系:WGS-84

时间动态变化的数据(速度,天气) 按照小时划分

路网数据, 包含路组成点的graph + 每条路的经纬度 + 路的长度

路段速度数据, 按照小时划分, 预测时刻前12小时, 前一天对应时刻的前12小时, 前一周对应时刻的前12小时

天气数据,所有路段共享,按照小时划分

POI数据,空间数据

交通事故数据

#数据预处理
遍历所有交通事故点:
1. 判断事故点是否在grids中,若不在,跳过
2. 找到距离事故点最近的node
3. 判断node是否在grids中,若不在,跳过
4. 找到node的50阶近邻,判断是否均在grids中,若不在,跳过,若在,生成node的道路信息(num_nodes, length) (roads)
5. 找到node和所有近邻的速度,若有缺失,填充,  前12小时,前一天对应时刻的前12小时, 前一周对应时刻的前12小时(speed)
6. 找到node和所有近邻的poi特征(poi坐标转换) (poi)
7. 找到node和所有近邻的天气特征 (weather)
8. 找到node和所有近邻的calendar特征 (calendar)
