import pandas as pd
from transform_coord.coord_converter import convert_by_type


longitudeMin = 116.09608
longitudeMax = 116.71040
latitudeMin = 39.69086
latitudeMax = 40.17647


def point_is_in_girds(longitude, latitude):
    return longitudeMin <= longitude <= longitudeMax and latitudeMin <= latitude <= latitudeMax


def clean_pois(POIFilePath, outRoadsPOIPath):
    origin_pois = pd.read_csv(POIFilePath, header=0)
    index_list = []
    for index, row in origin_pois.iterrows():
        longitude = row["LON"]
        latitude = row["LAT"]
        if not point_is_in_girds(longitude=longitude, latitude=latitude):
            print(f"ignore poi {index}")
            continue
        index_list.append(index)
    pois = origin_pois.iloc[index_list][["LON", "LAT", "TYPE_NUMBER"]].reset_index(drop=True)
    pois.columns = ["longitude", "latitude", "poi_type"]
    # convert longitude and latitude
    coords = list(zip(pois["longitude"].values.tolist(), pois["latitude"].values.tolist()))
    convert_corrds = []
    for coord in coords:
        convert_lng, convert_lat = convert_by_type(lng=coord[0], lat=coord[1], type="g2w")
        convert_corrds.append([convert_lng, convert_lat])
    pois["longitude"] = pd.Series(list(zip(*convert_corrds))[0])
    pois["latitude"] = pd.Series(list(zip(*convert_corrds))[1])
    pois.to_csv(outRoadsPOIPath, index=False)
    print(outRoadsPOIPath, " writes successfully.")


if __name__ == "__main__":
    POIFilePath = "../original_data/POI_data/poi_analyse.csv"
    outRoadsPOIPath = "../data/poi.csv"
    clean_pois(POIFilePath, outRoadsPOIPath)
