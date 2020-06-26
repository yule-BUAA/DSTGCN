import requests
import json
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

# data format :
# observation time : today 0:00:00 ~ today 23:30:00

attributeList = ["valid_time_gmt", "valid_time", "day_ind", "temp", "wx_icon", "icon_extd", "wx_phrase",
                 "dewPt", "heat_index", "rh", "pressure", "vis", "wc", "wdir", "wspd", "wdir_cardinal", "feels_like"]


# get feature from https://www.wunderground.com/
def getFeatureRawData():
    baseUrl = "https://api.weather.com/v1/geocode"
    # day dictionary
    monthDic = {
        "08": {"begin": "01", "end": "31"},
        "09": {"begin": "01", "end": "30"},
        "10": {"begin": "01", "end": "31"}
    }
    stationList = ["BeijingCapitalStation"]
    # longitude and latitude dictionary
    stationLatLongDic = {
        "BeijingCapitalStation": {"latitude": "40.07500076", "longitude": "116.58999634"},
    }
    # request parameters
    parameters = {
        "apiKey": "6532d6454b8aa370768e63d6ba5a832e",
        "startDate": "20190101",
        "endDate": "20190111",
        "units": "e"
    }
    year = "2018"
    basePath = "/home/yule/文档/weatherData/"
    for station in stationList:
        # print(station)
        latitude = stationLatLongDic[station]["latitude"]
        longitude = stationLatLongDic[station]["longitude"]
        for key in monthDic:
            url = baseUrl + "/" + latitude + "/" + longitude + "/observations/historical.json"
            # extract 2018 weather data
            parameters["startDate"] = year + key + monthDic[key]["begin"]
            parameters["endDate"] = year + key + monthDic[key]["end"]
            response = requests.get(url, params=parameters)
            print("request url is ", response.url)
            filePath = basePath + station + "_" + parameters["startDate"] + "-" + parameters["endDate"] + ".json"
            writeDataToJson(filePath, response)


# write raw weather data to json
def writeDataToJson(filePath, response):
    with open(filePath, "w") as f:
        # print(type(response.json()))
        json.dump(response.json(), f)
        print("data from " + response.url + " has been written.")


# vonvert Unix timestamp to local time
def convertTimestampToDatetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    # value stands for Unix timestamp, e.g. 1332888820
    value = time.localtime(value)
    # convert to specific format by using strftime function
    dt = time.strftime(format, value)
    return dt


# read raw weather json files and convert them into CSV files
def convertJsonToCSV(baseFilePath):
    # root -> 当前目录路径, dirs -> 当前路径下所有子目录, files -> 当前路径下所有非目录子文件
    for root, dirs, files in os.walk(baseFilePath):
        # print(files)
        for file in files:
            if not file.endswith(".json"):
                continue
            filePath = root + "/" + file
            with open(filePath, "r") as f:
                print(filePath)
                listDict = {}
                # fileRawData's type is a dictionary
                fileRawData = json.load(f)
                # observationData is a list containing json data
                observationData = fileRawData['observations']
                addToDict(listDict, observationData)
                # listDict -> {key : listname , value : list}
                # print(listDict)
                dataframe = pd.DataFrame({
                    attribute: listDict[attribute] for attribute in attributeList if attribute in listDict
                })
                # write into csv file, columns are ordered by attributeList
                dataframe.to_csv(filePath.replace(".json", ".csv"), index=False, columns=attributeList)
                print(file + " data length is " + str(len(fileRawData['observations'])))
                print(file.replace(".json", ".csv") + " has been written successfully.")


# add useful information in observationData into listDict
def addToDict(listDict, observationData):
    # fileRawData's type is a dictionary
    for data in observationData:
        for key in data:
            # print(key)
            if (key in attributeList):
                # check dict is empty or not
                if (key not in listDict):
                    listDict[key] = []
                listDict[key].append(data[key])
                # add convert time format
                if (key == "valid_time_gmt"):
                    if ("valid_time" not in listDict):
                        listDict["valid_time"] = []
                    listDict["valid_time"].append(convertTimestampToDatetime(data[key]))


# check which day's weather data is missing
def checkCSVData(baseFilePath):
    for root, dirs, files in os.walk(baseFilePath):
        # print(files)
        for file in files:
            if not file.endswith(".csv"):
                continue
            filePath = root + "/" + file
            data = pd.read_csv(filePath)
            correctCount = 48
            data['valid_time'] = pd.to_datetime(data['valid_time'])
            print("In ", file, " missing days are : ")
            for name, value in data.groupby(data['valid_time'].dt.day):
                if (len(value) != correctCount):
                    print("day %s, missing %d values " % (name, correctCount - len(value)))


# read weather data CSV
def readWeatherDataCSV(baseFilePath):
    for root, dirs, files in os.walk(baseFilePath):
        # print(files)
        for file in files:
            if not file.endswith(".csv"):
                continue
            filePath = root + "/" + file
            weatherData = pd.read_csv(filePath)
            columns = weatherData.columns.values
            # ['valid_time_gmt' 'valid_time' 'day_ind' 'temp' 'wx_icon' 'icon_extd'
            #                 'wx_phrase' 'dewPt' 'heat_index' 'rh' 'pressure' 'vis' 'wc' 'wdir' 'wspd']
            # print(columns)
            print(filePath, " data size is %d" % len(weatherData))
            timeList = weatherData['valid_time'].unique()
            print("total time records are %d" % len(timeList))


# plot weather data
def plotWeatherData(attribute):
    AugWeatherData = pd.read_csv("/home/yule/文档/weatherData/cleanedData/BeijingCapitalStation_20180801-20180831.csv")
    SepWeatherData = pd.read_csv("/home/yule/文档/weatherData/cleanedData/BeijingCapitalStation_20180901-20180930.csv")
    OctWeatherData = pd.read_csv("/home/yule/文档/weatherData/cleanedData/BeijingCapitalStation_20181001-20181031.csv")
    weatherDataList = [AugWeatherData, SepWeatherData, OctWeatherData]
    fig = plt.figure()
    ax_Aug = fig.add_subplot(1, 3, 1)
    ax_Sep = fig.add_subplot(1, 3, 2)
    ax_Oct = fig.add_subplot(1, 3, 3)

    for index, data in enumerate(weatherDataList):
        ax = None
        data["valid_time"] = pd.to_datetime(data["valid_time"])
        if index == 0:
            ax = ax_Aug
        elif index == 1:
            ax = ax_Sep
        else:
            ax = ax_Oct
        # print(type(data.groupby([data["valid_time"].dt.day, data["valid_time"].dt.hour])))
        group = data.groupby([data["valid_time"].dt.day, data["valid_time"].dt.hour])[attribute]
        groupMean = group.mean().unstack().transpose()
        # print(groupMean)
        groupMean.plot(ax=ax, kind="line", figsize=(15, 15), title='month ' + str(index + 8), legend=False)
    plt.show()


def make_external_features(weather_path: str):
    # rh 湿度, wdir_cardinal 风向
    filePathList = ["/home/yule/文档/weatherData/BeijingCapitalStation_20180801-20180831.csv",
                    "/home/yule/文档/weatherData/BeijingCapitalStation_20180901-20180930.csv",
                    "/home/yule/文档/weatherData/BeijingCapitalStation_20181001-20181031.csv"
                    ]
    dataList = [pd.read_csv(path, usecols=['temp', 'dewPt', 'rh', 'wdir_cardinal', 'wspd', 'pressure', 'wx_phrase',
                                           'valid_time', 'feels_like']) for path in filePathList]
    weather = pd.concat([data for data in dataList])
    weather.fillna(method='ffill', axis=0, inplace=True)
    weather["valid_time"] = pd.to_datetime(weather["valid_time"])
    weather = weather.set_index('valid_time')

    # 剔除该列里的多余值
    weather['wx_phrase'] = weather['wx_phrase'].apply(lambda val: val.split('/')[0].strip())
    # one-hot encoding
    weather = pd.get_dummies(weather)

    weather = weather.resample('1H').mean()

    weather.fillna(method='ffill', axis=0, inplace=True)
    weather.to_csv(weather_path)
    print(weather_path, " writes successfully.")


if __name__ == '__main__':
    # excute for only once
    # getFeatureRawData()
    baseFilePath = "/home/yule/文档/weatherData"
    outWeatherCSVPath = "/home/yule/桌面/traffic_accident_data/weather.csv"
    #  generate csv files
    # convertJsonToCSV(baseFilePath)
    # checkCSVData(baseFilePath)
    # readWeatherDataCSV(baseFilePath)
    # plotWeatherData("temp")
    make_external_features(outWeatherCSVPath)
