import time
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
import shutil
import csv
from collections import defaultdict
import math
import multiprocessing
from tqdm import tqdm

divideBound = 5
file_count = 0
floatBitNumber = 8

longitudeMin = 116.09608
longitudeMax = 116.71040
latitudeMin = 39.69086
latitudeMax = 40.17647
# 网格的划分
widthSingle = 0.01 / math.cos(latitudeMin / 180 * math.pi) / divideBound
width = math.floor((longitudeMax - longitudeMin) / widthSingle)
heightSingle = 0.01 / divideBound
height = math.floor((latitudeMax - latitudeMin) / heightSingle)
print("height = ", height)
print("heightSingle = ", heightSingle)
print("width = ", width)
print("widthSingle = ", widthSingle)


# check date format
def isValidDate(date):
    try:
        time.strptime(date, "%Y-%m-%d %H:%M:%S")
        return True
    except:
        return False


def isZeroSpeed(speed):
    try:
        transfer = float(speed)
        if(transfer == 0):
            return True
        return False
    except:
        return True


# unzip 23990 files, file_count = 23990
def unzipFile(baseFilePath):
    global file_count
    files = os.listdir(baseFilePath) # get all File Names
    print("start unzip files in ", baseFilePath)
    for file in files:
        filePath = os.path.join(baseFilePath, file)
        if(os.path.isfile(filePath) and filePath.endswith(".zip")):
            zFile = zipfile.ZipFile(filePath, "r")
            # get all fileNames in zFile
            for fileM in zFile.namelist():
                zFile.extract(fileM, baseFilePath)
                file_count += 1
                print("unzip a file in ", baseFilePath)
            zFile.close();
        elif(os.path.isdir(filePath)):
            unzipFile(filePath)
        print(file + " unzip completed")
    # print(files)
    print("total unzip %d files." %file_count)


# delete all zip files
def deleteZipFiles(baseFilePath):
    files = os.listdir(baseFilePath) # get all File Names
    for file in files:
        filePath = os.path.join(baseFilePath, file)
        if(os.path.isfile(filePath) and filePath.endswith(".zip")):
            os.remove(filePath)
            print("delete ", filePath)
        elif(os.path.isdir(filePath)):
            deleteZipFiles(filePath)


# move files to parent dir
def moveFiles(baseFilePath):
    files = os.listdir(baseFilePath) # get all File Names
    print(files)
    for file in files:
        parentPath = os.path.join(baseFilePath, file)
        Files = os.listdir(parentPath)
        print(Files)
        assert (len(Files) == 1 and os.path.isdir(os.path.join(parentPath, Files[0])))
        secondFiles = os.listdir(os.path.join(parentPath, Files[0]))
        for secondFile in secondFiles:
            PATH = os.path.join(parentPath, Files[0], secondFile)
            if(os.path.isfile(PATH) and PATH.endswith(".txt")):
                shutil.move(PATH, os.path.join(parentPath, secondFile))
                print("move ", PATH, " -> ", os.path.join(parentPath, secondFile))
        # delete empty folder
        os.rmdir(os.path.join(parentPath, Files[0]))
        print("delete ", os.path.join(parentPath, Files[0]))


# count file numbers each day
# miss 490 files
def countFiles(baseFilePath):
    count = 0
    correctNumber = 288
    files = os.listdir(baseFilePath) # get all File Names
    for file in files:
        childFiles = os.listdir(os.path.join(baseFilePath, file))
        if(len(childFiles) != correctNumber):
            print(file, 'has %d files, missing %d files' %(len(childFiles), correctNumber - len(childFiles)))
            count = count + correctNumber - len(childFiles)
    print("missing ", count, " files in total.")


# read txt files
def readTXTFiles(baseFilePath):
    dailyFilePath = os.listdir(baseFilePath)
    attributeList = ['longitude', 'latitude', 'speed', 'angle_speed', 'time']
    index_list = [4, 5, 6, 7, 10]
    for dailyFile in dailyFilePath:
        PATH = os.path.join(baseFilePath, dailyFile)
        txt_filePath_list = os.listdir(PATH)
        listContent = []
        for txt_filePath in txt_filePath_list:
            print(txt_filePath)
            in_txt = csv.reader(open(os.path.join(baseFilePath, dailyFile, txt_filePath), "r"), delimiter=',', escapechar='\n')
            listContent.append(list(in_txt))
        # add daily information in one list
        dailyContent = [j for i in listContent for j in i]
        dic = {}
        for line in dailyContent:
            if (len(line) != 11):
                continue
            if(not isValidDate(line[10][:-1])):
                continue
            for i, index in enumerate(index_list):
                if (attributeList[i] not in dic):
                    dic[attributeList[i]] = []
                # remove last position which is semicolon;
                if (i == len(index_list) - 1):
                    line[index] = line[index][:-1]
                dic[attributeList[i]].append(line[index])
        dataFrame = pd.DataFrame(dic)
        dataFrame['time'] = pd.to_datetime(dataFrame['time'])
        # if problem called DataError: No numeric types to aggregate occurs, change data type first!!!!
        dataFrame['longitude'] = dataFrame['latitude'].astype('float32')
        dataFrame['latitude'] = dataFrame['latitude'].astype('float32')
        dataFrame['speed'] = dataFrame['speed'].astype('int')
        dataFrame['angle_speed'] = dataFrame['angle_speed'].astype('int')
        # data = dataFrame.groupby(pd.TimeGrouper(key='time', freq='30Min'))
        # remove 'time' column
        # print(data.mean())

        # divide into grids
        # grids[][]
        # for key, value in data:
        #     value = value.drop(columns=['time'])
        # print(data)
        # print(data.mean())
        x = dataFrame['longitude'].values
        y = dataFrame['latitude'].values

        x_list = []
        y_list = []
        for i, j in zip(x, y):
            if j >= 39.26 and j <= 41.03 and i >= 115.25 and i <= 117.30:
                x_list.append(i)
                y_list.append(j)
        plt.scatter(x_list, y_list)
        break


# 得到所有位置的信息,一次全部计算出,但是计算量太大,可行性不强
def getRoadSpeed(baseFilePath, roadMonitorLocationLongLatPath):

    # 在经线上，纬度每差1度,实地距离大约为111千米；
    # 在纬线上，经度每差1度, 实际距离为111×cosθ千米。（其中θ表示该纬线的纬度.在不同纬线上, 经度每差1度的实际距离是不相等的）。
    # 选取的距离监控路段的距离为1110m内的poi, 南北走向 1110m 相当于 1 / 100 度, 东西走向1110m 相当于 1 / (100 * cosθ)
    roadMonitorLocationDataFrame = pd.read_csv(roadMonitorLocationLongLatPath)
    locations = [(roadMonitorLocationDataFrame.iloc[i][0], roadMonitorLocationDataFrame.iloc[i][1]) for i in range(len(roadMonitorLocationDataFrame))]
    # 对每天的出租车GPS数据进行读取,放入一个dataFrame里面,然后对于这个dataFrame循环所有的location,筛选出在范围内的出租车,之后resample,再求speed的平均值
    dailyFilePath = os.listdir(baseFilePath)
    attributeList = ['longitude', 'latitude', 'speed', 'angle_speed', 'time']
    index_list = [4, 5, 6, 7, 10]
    # allLocationSpeedDataFrame = pd.DataFrame()
    for dailyFile in dailyFilePath:
        PATH = os.path.join(baseFilePath, dailyFile)
        txt_filePath_list = os.listdir(PATH)
        # 将一天的所有数据读入到listContent中
        listContent = []
        for txt_filePath in txt_filePath_list:
            print(txt_filePath)
            in_txt = csv.reader(open(os.path.join(baseFilePath, dailyFile, txt_filePath), "r"), delimiter=',',
                                escapechar='\n')
            listContent.append(list(in_txt))
        # add daily information in one list
        dailyContent = [minuteStep for hourAll in listContent for minuteStep in hourAll]
        print("dailyContent length is ", len(dailyContent))
        dailyDic = defaultdict(list)
        for line in dailyContent:
            if (len(line) != 11):
                continue
            # 过滤所有速度为0的点
            # if (isZeroSpeed(line[6])):
            #     continue
            if (not isValidDate(line[10][:-1])):
                continue
            for i, index in enumerate(index_list):
                # remove ;
                if (i == len(index_list) - 1):
                    line[index] = line[index][:-1]
                dailyDic[attributeList[i]].append(line[index])
        print("remove 0 speed, dailyContent length is ", len(dailyDic["speed"]))
        dailyDataFrame = pd.DataFrame(dailyDic)
        dailyDataFrame["longitude"] = dailyDataFrame["longitude"].astype(float)
        dailyDataFrame["latitude"] = dailyDataFrame["latitude"].astype(float)
        dailyDataFrame["speed"] = dailyDataFrame["speed"].astype(float)

        allLocationDailyDataFrameDic = {}
        dealLocationCount = 0
        for location in locations:
            longitude = location[0]
            latitude = location[1]
            # 1110m 划分为 divideBound 块,然后再按照一半的距离再次划分块(类似于圆形区域)
            divideNum = divideBound * 2
            locationData = dailyDataFrame[dailyDataFrame.apply(lambda x : math.fabs(x["longitude"] - longitude)
                            <= math.fabs(0.01 / math.cos(x["latitude"] / 180 * math.pi)) / divideNum and math.fabs(x["latitude"] - latitude) <= 0.01 / divideNum, axis = 1)]
            print("->", locationData)
            # 一个符合条件的点都没有
            if(len(locationData) == 0):
                allLocationDailyDataFrameDic[str(location[0]) + "," + str(location[1])] = [0] * 24
            locationData["time"] = pd.to_datetime(locationData["time"])
            locationData = locationData.set_index("time")
            locationSpeedMeanSeries = locationData.resample(rule="1H")["speed"].mean()
            # for ind in locationSpeedMeanSeries.index:
            #     print(ind, " -> ", locationSpeedMeanSeries[index])
            allLocationDailyDataFrameDic[str(location[0]) + "," + str(location[1])] = locationSpeedMeanSeries
            dealLocationCount += 1
            print("has dealt %d locations" %dealLocationCount)
        allLocationDailyDataFrame = pd.concat(allLocationDailyDataFrameDic, axis=1).fillna(0)
        # 单个文件输出
        allLocationDailyDataFrame.to_csv(PATH + "/" + dailyFile + "_Speed.csv")
        # allLocationSpeedDataFrame = pd.concat([allLocationSpeedDataFrame, allLocationDailyDataFrame], axis=0)
    # allLocationSpeedDataFrame = allLocationSpeedDataFrame.fillna(0)
    # outlocationSpeedPath = baseFilePath + "/allRoadsSpeed.csv"
    # allLocationSpeedDataFrame.to_csv(outlocationSpeedPath)
    # print(outlocationSpeedPath, " writes successfully.")

# 计算道路的速度,思路:对于监控总区域划分网格,220m*220m,
# 然后将每一天的出租车数据进行网格上的划分,得到每个网格按照时间 1H 分隔的Series, 每天一共的数量是12 * (m * n)
# 最后对于每个道路监控点,定位到其网格,然后把网格该天的Series赋值个给这个监控点


# 得到每个网格的速度
def getGridTaxiSpeed(baseFilePath, filePathList):
    # 在经线上，纬度每差1度,实地距离大约为111千米；
    # 在纬线上，经度每差1度, 实际距离为111×cosθ千米。（其中θ表示该纬线的纬度.在不同纬线上, 经度每差1度的实际距离是不相等的）。
    # 选取的距离监控路段的距离为1110m内的poi, 南北走向 1110m 相当于 1 / 100 度 0.01度, 东西走向1110m 相当于 1 / (100 * cosθ), 0.01 / cosθ
    # 对每天的出租车GPS数据进行读取,去掉在划分的网格区域外的数据, 之后计算划分的网格每天的速度
    attributeList = ['longitude', 'latitude', 'speed', 'time']
    index_list = [4, 5, 6, 10]
    # 读取每天的出租车数据文件
    for dailyFile in filePathList:
        PATH = os.path.join(baseFilePath, dailyFile)
        txt_filePath_list = os.listdir(PATH)
        # 将一天的所有数据读入到listContent中
        listContent = []
        for txt_filePath in txt_filePath_list:
            print(txt_filePath)
            in_txt = csv.reader(open(os.path.join(baseFilePath, dailyFile, txt_filePath), "r"), delimiter=',',
                                escapechar='\n')
            listContent.append(list(in_txt))
        # add daily information in one list
        dailyContent = [minuteStep for hourAll in listContent for minuteStep in hourAll]
        print("dailyContent length is ", len(dailyContent))
        dailyDic = defaultdict(list)
        for line in dailyContent:
            if (len(line) != 11):
                continue
            # 过滤所有速度为0的点
            # if (isZeroSpeed(line[6])):
            #     continue
            if (not isValidDate(line[10][:-1])):
                continue
            for i, index in enumerate(index_list):
                # remove last position semicolon ;
                if (i == len(index_list) - 1):
                    line[index] = line[index][:-1]
                dailyDic[attributeList[i]].append(line[index])
        print("remove 0 speed, dailyContent length is ", len(dailyDic["speed"]))
        dailyDataFrame = pd.DataFrame(dailyDic)
        dailyDataFrame["longitude"] = dailyDataFrame["longitude"].astype(float)
        dailyDataFrame["latitude"] = dailyDataFrame["latitude"].astype(float)
        dailyDataFrame["speed"] = dailyDataFrame["speed"].astype(float)
        # 去掉在划分的网格区域外的数据
        dailyDataFrame = dailyDataFrame[dailyDataFrame.apply(lambda x : x["longitude"] >= longitudeMin and x["longitude"] <= longitudeMax and x["latitude"] >= latitudeMin
                                                         and x["latitude"] <= latitudeMax, axis=1)]
        print("remove not in grids range taxi data, dailyContent length is ", len(dailyDataFrame))
        allGridsTaxiData = [[defaultdict(list) for j in range(width + 1)] for i in range(height + 1)]
        print("generate allGridsTaxiData.")
        print("dailyDataFrame length is ", len(dailyDataFrame))
        for _, row in dailyDataFrame.iterrows():
            widthIndex = math.floor((row["longitude"] - longitudeMin) / widthSingle)
            heightIndex = math.floor((row["latitude"] - latitudeMin) / heightSingle)
            # print("widthIndex = ", widthIndex)
            # print("heightIndex = ", heightIndex)
            assert (widthIndex <= width)
            assert (heightIndex <= height)
            allGridsTaxiData[heightIndex][widthIndex]["longitude"].append(row["longitude"])
            allGridsTaxiData[heightIndex][widthIndex]["latitude"].append(row["latitude"])
            allGridsTaxiData[heightIndex][widthIndex]["speed"].append(row["speed"])
            allGridsTaxiData[heightIndex][widthIndex]["time"].append(row["time"])
        print("dailyDataFrame has divided into grids.")
        # 存储所有网格的速度Series序列
        allGridsSpeedDic = {}
        timeRange = pd.date_range(dailyFile, periods=24, freq="1H")
        for row in range(height + 1):
            for column in range(width + 1):
                print("deal with %d row and %d column" %(row, column))
                gridDataFrame = pd.DataFrame(allGridsTaxiData[row][column])
                # 网格中该天一个出租车数据点都没有
                if(len(gridDataFrame) == 0):
                    allGridsSpeedDic[str(row) + "," + str(column)] = [0] * 24
                else:
                    gridDataFrame["time"] = pd.to_datetime(gridDataFrame["time"])
                    gridDataFrame = gridDataFrame.set_index("time")
                    gridSpeedSeries = gridDataFrame.resample(rule="1H")["speed"].mean().fillna(0)
                    allGridsSpeedDic[str(row) + "," + str(column)] = [gridSpeedSeries.get(eachTime, default=0.0) for eachTime in timeRange]
        pd.DataFrame(allGridsSpeedDic, index=timeRange).to_csv(PATH + "/gridsSpeed.csv")
        print(PATH + "/gridsSpeed.csv writes successfully.")


# 得到每个网格的出租车速度,使用多线程方法
def getGridTaxiSpeedMultiKernel(baseFilePath, kernel = 16):

    dailyFilePathList = os.listdir(baseFilePath)
    eachKernelFileCount = int(math.ceil(len(dailyFilePathList) / kernel))

    pool = multiprocessing.Pool(processes=kernel)

    for i in range(0, len(dailyFilePathList), eachKernelFileCount):
        endIndex = i + eachKernelFileCount
        if(endIndex > len(dailyFilePathList)):
            endIndex = len(dailyFilePathList)
        pool.apply_async(getGridTaxiSpeed, (baseFilePath, dailyFilePathList[i : endIndex]))
    pool.close()
    pool.join()


def merge_all_grids_speed(baseFilePath):

    all_grids_speed_dataframe = pd.DataFrame()
    dailyFilePathList = os.listdir(baseFilePath)
    # 查找每天文件夹下的girdsSpeed.csv文件
    for dailyFile in tqdm(dailyFilePathList):
        PATH = os.path.join(baseFilePath, dailyFile)
        if not os.path.isdir(PATH):
            continue
        gridsSpeedDataFrame = pd.read_csv(os.path.join(PATH, "gridsSpeed.csv"), index_col=0, parse_dates=True)
        all_grids_speed_dataframe = pd.concat([all_grids_speed_dataframe, gridsSpeedDataFrame], axis=0)

    all_grids_speed_dataframe.to_csv(baseFilePath + "/all_grids_speed.csv")
    print(baseFilePath + "/all_grids_speed.csv writes successfully.")


def getGridPoints():
    longitudeList =[]
    latitudeList = []
    for row in range(height + 1):
        for column in range(width + 1):
            longitudeList.append(longitudeMin + column * widthSingle)
            latitudeList.append(latitudeMin + row * heightSingle)
    pd.DataFrame({
        "longitude": longitudeList,
        "latitude": latitudeList
    }).to_csv("/home/yule/文档/gridPoints.csv")


def move_grids_speed_files(baseFilePath):
    new_base_path = "F:\\beijing_grids_speed"
    if not os.path.exists(new_base_path):
        os.makedirs(new_base_path)
    dailyFilePathList = os.listdir(baseFilePath)
    for dailyFile in dailyFilePathList:
        PATH = os.path.join(baseFilePath, dailyFile)
        if not os.path.isdir(PATH):
            continue
        new_file_folder_path = os.path.join(new_base_path, dailyFile)
        if not os.path.exists(new_file_folder_path):
            os.makedirs(new_file_folder_path)
        shutil.copy(os.path.join(PATH, "gridsSpeed.csv"), os.path.join(new_file_folder_path, "gridsSpeed.csv"))


if __name__ == "__main__":
    baseFilePath = "/home/yule/文档/BeijingTaxi/data"
    # roadAccidentCountCSVPath = "/home/yule/文档/accident/roadAccidentCount.csv"
    inputRoadMonitorLocationLongLatPath = "/home/yule/文档/accident/SelectedRoadPoints.csv"

    # unzipFile(baseFilePath)
    # deleteZipFiles(baseFilePath)
    # moveFiles(baseFilePath)
    # countFiles(baseFilePath)
    # readTXTFiles(baseFilePath)

    # getGridTaxiSpeed(baseFilePath, os.listdir(baseFilePath))
    # getGridTaxiSpeedMultiKernel(baseFilePath, 16)

    # getGridPoints()
    # move_grids_speed_files(baseFilePath)
    merge_all_grids_speed("/home/yule/桌面/traffic_accident_data/beijing_grids_speed")
