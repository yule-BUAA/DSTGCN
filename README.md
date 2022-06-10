# Deep Spatio-Temporal Graph Convolutional Network for Traffic Accident Prediction (DSTGCN)

DSTGCN is a graph-based neural network that predicts the risk of traffic accidents in the future.

Please refer to our Neurocomputing 2021 paper [“**Deep Spatio-Temporal Graph Convolutional Network for Traffic Accident Prediction**”](https://www.sciencedirect.com/science/article/pii/S092523122031451X#!) for more details.

## Project Structure
The descriptions of principal files in this project are introduced as follows:
- model/
    - `spatial_layer.py`: the Spatial Convolutional Layer
    - `spatial_temporal_layer.py`: the Spatial-Temporal Convolutional Layer
    - `fully_connected.py`: fully connected network for the Embedding layer
    - `DSTGCN.py`: the Deep Spatio-Temporal Graph Convolutional Network
- preprocessing_data/
  - `process_xxx.py`: process the raw 'xxx' dataset
  - `generate_data.py`: generate the dataset for our model
- transform_coord/: convert the coordinate, which could be obtained from [here](https://github.com/wandergis/coordTransform_py).
- train/
  - `train_model.py` and `train_main.py`: train models
- test/
  - `test_main.py`: evaluate models
- utils/: utility files (e.g. data loader, metrics calculation, loss function, configurations) 
- original_data/: original datasets. Due to the data privacy, we do not provide the original data. But, you could use the preprocess codes in preprocessing_data/ folder to generate your own datasets and use them to train the model. 
- data/: processed datasets. We provide a sampled dataset [here](https://drive.google.com/file/d/1MQTzu_NqzeRzQ7jc2MEdKU3agvul_wJF/view?usp=sharing). You can download it and then put the data files in this folder.
- saves/ and runs/: folders to save models and outputs of tensorboardX, respectively
- results/: folders to save the evaluation metrics for models. 

## Format of the processed data:
- accident data format: records of traffic accidents.

| "longitude" | "latitude" | "startTime" | "endTime" |
|  ----  | ----  | ----  | ----  | 
| accident longitude | accident latitude | accident start time | accident end time | 

- poi data format: records of pois.

| "longitude" | "latitude" | "poi_type" |
| ----  | ----  | ----  | 
| poi longitude | poi latitude | poi function type | 

- road data (beijing_roadnet.gpickle) format:
networkx.classes.graph.Graph, road network structure that records the connectivity of road segments.
You can download the original data from [here](https://figshare.com/articles/dataset/Urban_Road_Network_Data/2061897), 
and then run preprocessing_data/`process_beijing_road_net.py` to get the preprocessed file. 

- speed data (all_grids_speed.h5) format: DataFrame, containing the traffic speed of each grid

- weather data format (weather.h5) format: records of the weather condition.
You can run preprocessing_data/`process_weather.py` to get the preprocessed file.
  
| "temp" | "dewPt" | "pressure" | "wspd" | ... |
| ----  | ----  | ----  |  ----  | ----  | 
| temperature | dew point | pressure | wind speed | etc. | 
 
- edge.h5 data format: records of spatial features (poi and road segment features), 
  which is a combination of preprocessed poi data and road data.

| "XCoord" | "YCoord" | "LENGTH" | "NUM_NODE" | "spatial_features" |
| ----  | ----  | ----  |  ----  | ----  | 
| road segment longitude | road segment latitude | road segment length | points that road segment contains | road segment poi distribution (a list of each poi type numbers) | 


## Parameter Settings
Please refer to our paper for more details of parameter settings. 
Hyperparameters could be found in utils/`config.json` and you can adjust them when running the model.

## How to use
- Training: after setting the parameters, run ```python train_main.py``` to train models. 
- Testing: run ```python test_main.py``` to evaluate models based on the path of saved models. 

##  Principal environmental dependencies
- [PyTorch == 1.8.1](https://pytorch.org/)
- [dgl == 0.7.0](https://www.dgl.ai/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [sklearn](https://scikit-learn.org/stable/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)

## Citation
Please consider citing the following paper when using our data or code.

```bibtex
@article{DBLP:journals/ijon/YuDHSHL21,
  author    = {Le Yu and
               Bowen Du and
               Xiao Hu and
               Leilei Sun and
               Liangzhe Han and
               Weifeng Lv},
  title     = {Deep spatio-temporal graph convolutional network for traffic accident
               prediction},
  journal   = {Neurocomputing},
  volume    = {423},
  pages     = {135--147},
  year      = {2021}
}
```
