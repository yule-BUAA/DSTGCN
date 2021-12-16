import time
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from transform_coord.coord_converter import convert_by_type


# check float format
def isFloat(number):
    try:
        float(number)
        return True
    except ValueError:
        return False


# check date format
def isValidDate(date):
    try:
        time.strptime(date, "%Y/%m/%d %H:%M:%S")
        return True
    except:
        return False


# transfer txt to csv
def get_accident(filePath, outFilePath):
    in_txt = csv.reader(open(os.path.join(filePath), "r"), delimiter=';', escapechar='\n')
    dataList = list(in_txt)
    count = 0
    dic = defaultdict(list)
    for data in dataList:
        if len(data) <= 2:
            continue
        isAccident = False
        # filter accident data
        for info in data:
            if "事故" in str(info):
                isAccident = True
                break
        if(isAccident):
            location = data[1]
            last = data[-1]
            longitude, latitude = location.split(",")
            if not isFloat(longitude) or not isFloat(latitude):
                continue
            splitList = last.split('\t')
            startTime, endTime = splitList[-2], splitList[-1]
            if not isValidDate(startTime) or not isValidDate(endTime):
                continue

            longitude, latitude = convert_by_type(lng=float(longitude), lat=float(latitude), type='g2w')

            dic["longitude"].append(longitude)
            dic["latitude"].append(latitude)
            dic["startTime"].append(startTime)
            dic["endTime"].append(endTime)
            count += 1
    print("accident data counts : ", count)
    dataFrame = pd.DataFrame(dic)
    dataFrame.to_csv(outFilePath, index=False, columns=["longitude", "latitude", "startTime", "endTime"])
    print(outFilePath, " writes successfully.")


# plot accident longitude and latitude scatter
def plot_accident_location(filePath):
    dataFrame = pd.read_csv(filePath)
    plt.scatter(dataFrame['longitude'], dataFrame['latitude'])
    plt.show()


# plot daily / hourly  accident numbers
def plot_accient_num(filePath):
    dataFrame = pd.read_csv(filePath)
    # columns = ['longitude', 'latitude', 'startTime', 'endTime']
    dataFrame['startTime'] = pd.to_datetime(dataFrame['startTime'])
    dataFrame['endTime'] = pd.to_datetime(dataFrame['endTime'])
    hourGroup = dataFrame.groupby(pd.Grouper(key='endTime', freq="1H"))["longitude"]
    hourGroupMean = hourGroup.count()
    plt.title('hourly accidents')
    plt.ylabel('traffic accident numbers')
    hourGroupMean.plot()
    plt.show()


if __name__ == "__main__":
    inFilePath = "../original_data/accident_data/event.txt"
    outFilePath = "../data/accident.csv"
    get_accident(inFilePath, outFilePath)
    plot_accient_num(outFilePath)
    plot_accident_location(outFilePath)
