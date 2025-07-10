import os, random, json, requests, folium
import pandas as pd
import seaborn as sns 
import scipy as sp
import numpy as np
import branca.colormap as cm
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
from scipy import interpolate
from scipy.interpolate import interp1d,UnivariateSpline
from scipy.stats import *
from folium.plugins import HeatMap
from geopy.distance import geodesic #몇개인지 세 주는 모듈
# rc('font', family = 'AppleGothic') #실행환경이 Mac일 경우
os.chdir(os.path.dirname(__file__))


tai = input('tai? : 예) 붕어빵, 스타벅스 등등...')
def getTaiCoordsDf(tai=str):
    if tai =='붕어빵':
        tai_coords = pd.read_excel('/Volumes/YAHO_/002. 데이터(자료 백업용)/붕어빵 2025.01.01~/INPUTS_update_0417/20250203_dagn.xlsx') #붕어빵 좌표
    else:
        try:
            tai_coords = pd.read_excel(f'/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/{tai}_XY.xlsx')
        except FileNotFoundError:
            # tai_coords = pd.read_excel(f'/Volumes/YAHO_/000. Processing Python/붕어빵/Inputs/{tai}_XY.xlsx')
            print(f'FileNotFoundError : 파일 경로 확인 필요 - {tai}_XY.xlsx')
    return tai_coords
# tai_coords = pd.read_excel('/Volumes/YAHO_/000. Processing Python/붕어빵/Inputs/20250203_dagn.xlsx')
# tai_coords = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/스타벅스_XY.xlsx')

tai_coords = getTaiCoordsDf(tai) # 붕어빵 좌표를 불러온다
station_coords = pd.read_excel('/Volumes/YAHO_/999.용도별 코드_Input포함/20250205_서울시내 역(station_df, cent_df 통합).xlsx')
def all_count_tai(coords_df, filename, radius_meters_list):# 모든 역에 대하여 일정 거리 이내의 붕어빵 좌표 개수를 엑셀로 저장
    station_labels, radius_labels = coords_df['역사명'], radius_meters_list
    count_matrix = np.zeros((len(station_labels), len(radius_labels)))
    for i,(_, station) in enumerate(coords_df.iterrows()):
        print(f'[{i+1}/305]개 역 계산중...')
        for j, radius in enumerate(radius_labels):
            count_all = len([tai for tai in tai_coords.itertuples() if geodesic((station['위도'],station['경도']),
                                                                                   (tai.Longitude,tai.Latitude)).meters <= radius])
            count_matrix[i,j] = count_all
    df_count_all = pd.DataFrame(count_matrix, index = station_labels, columns = radius_meters_list)
    df_count_all.to_excel(f'{filename}_all_count_tai.xlsx')

def all_count_station(coords_df, filename, radius_meters_list):# 모든 역에 대하여 일정 거리 이내의 붕어빵 좌표 개수를 엑셀로 저장
    station_labels, radius_labels = coords_df['역사명'], radius_meters_list
    count_matrix = np.zeros((len(station_labels), len(radius_labels)))
    for i,(_, station) in enumerate(coords_df.iterrows()):
        print(f'[{i+1}/305]개 역 계산중...')
        for j, radius in enumerate(radius_labels):
            count_all = len([stations for stations in coords_df.itertuples() if geodesic((station['위도'],station['경도']),
                                                                                   (stations.위도,stations.경도)).meters <= radius])
            count_matrix[i,j] = count_all-1 #자기자신제외
    df_count_all = pd.DataFrame(count_matrix, index = station_labels, columns = radius_meters_list)
    df_count_all.to_excel(f'{filename}_all_count_station.xlsx')

# all_count_tai(station_coords, '붕어빵_300_2500', np.arange(300, 2510, 100))
all_count_tai(station_coords, f'{tai}_100_3000', np.arange(100 , 3010, 100))
# all_count_station(station_coords, '역간 거리당 역개수_300_2500', np.arange(300, 2510, 100))