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
# divide grids
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
            zFile.close()
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

# get each grid speed
def getGridTaxiSpeed(baseFilePath, filePathList):

    attributeList = ['longitude', 'latitude', 'speed', 'time']
    index_list = [4, 5, 6, 10]

    for dailyFile in filePathList:
        PATH = os.path.join(baseFilePath, dailyFile)
        txt_filePath_list = os.listdir(PATH)
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
        # remove data that out of grid
        dailyDataFrame = dailyDataFrame[dailyDataFrame.apply(lambda x : x["longitude"] >= longitudeMin and x["longitude"] <= longitudeMax and x["latitude"] >= latitudeMin
                                                         and x["latitude"] <= latitudeMax, axis=1)]
        print("remove not in grids range taxi data, dailyContent length is ", len(dailyDataFrame))
        allGridsTaxiData = [[defaultdict(list) for j in range(width + 1)] for i in range(height + 1)]
        print("generate allGridsTaxiData.")
        print("dailyDataFrame length is ", len(dailyDataFrame))
        for _, row in dailyDataFrame.iterrows():
            widthIndex = math.floor((row["longitude"] - longitudeMin) / widthSingle)
            heightIndex = math.floor((row["latitude"] - latitudeMin) / heightSingle)

            assert (widthIndex <= width)
            assert (heightIndex <= height)
            allGridsTaxiData[heightIndex][widthIndex]["longitude"].append(row["longitude"])
            allGridsTaxiData[heightIndex][widthIndex]["latitude"].append(row["latitude"])
            allGridsTaxiData[heightIndex][widthIndex]["speed"].append(row["speed"])
            allGridsTaxiData[heightIndex][widthIndex]["time"].append(row["time"])
        print("dailyDataFrame has divided into grids.")

        allGridsSpeedDic = {}
        timeRange = pd.date_range(dailyFile, periods=24, freq="1H")
        for row in range(height + 1):
            for column in range(width + 1):
                print("deal with %d row and %d column" %(row, column))
                gridDataFrame = pd.DataFrame(allGridsTaxiData[row][column])

                if(len(gridDataFrame) == 0):
                    allGridsSpeedDic[str(row) + "," + str(column)] = [0] * 24
                else:
                    gridDataFrame["time"] = pd.to_datetime(gridDataFrame["time"])
                    gridDataFrame = gridDataFrame.set_index("time")
                    gridSpeedSeries = gridDataFrame.resample(rule="1H")["speed"].mean().fillna(0)
                    allGridsSpeedDic[str(row) + "," + str(column)] = [gridSpeedSeries.get(eachTime, default=0.0) for eachTime in timeRange]
        pd.DataFrame(allGridsSpeedDic, index=timeRange).to_csv(PATH + "/gridsSpeed.csv")
        print(PATH + "/gridsSpeed.csv writes successfully.")


def getGridTaxiSpeedMultiKernel(baseFilePath, kernel=16):

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


def move_grids_speed_files(baseFilePath, new_base_path):
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


def merge_all_grids_speed(new_base_path):

    all_grids_speed_dataframe = pd.DataFrame()
    dailyFilePathList = os.listdir(new_base_path)

    for dailyFile in tqdm(dailyFilePathList):
        PATH = os.path.join(new_base_path, dailyFile)
        if not os.path.isdir(PATH):
            continue
        gridsSpeedDataFrame = pd.read_csv(os.path.join(PATH, "gridsSpeed.csv"), index_col=0, parse_dates=True)
        all_grids_speed_dataframe = pd.concat([all_grids_speed_dataframe, gridsSpeedDataFrame], axis=0)

    # all_grids_speed_dataframe.to_csv("../data/all_grids_speed.csv")
    all_grids_speed_dataframe.to_hdf("../data/all_grids_speed.h5", key='key', mode='w')

    print("../data/all_grids_speed.csv writes successfully.")


if __name__ == "__main__":
    baseFilePath = "../original_data/BeijingTaxi/data"

    unzipFile(baseFilePath)
    deleteZipFiles(baseFilePath)
    moveFiles(baseFilePath)
    countFiles(baseFilePath)
    readTXTFiles(baseFilePath)

    # getGridTaxiSpeed(baseFilePath, os.listdir(baseFilePath))
    getGridTaxiSpeedMultiKernel(baseFilePath, 16)

    new_base_path = "../original_data/beijing_grids_speed"
    move_grids_speed_files(baseFilePath, new_base_path)
    merge_all_grids_speed(new_base_path)
