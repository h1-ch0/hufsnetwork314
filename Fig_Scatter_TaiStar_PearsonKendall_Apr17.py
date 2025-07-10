'''
1.반경을 바꾸기 용이할 것
2. 역을 바꾸기 용이할 것
3. 후에 거리의 역수 개수를 가져와서 그리기 용이할 것'''

"""붕어빵과 스타벅스의 거리별 개수를 CC_x축, 개수_y축 Scatter
pearson, kendall 상관관계와 p-value
총 count된 판매점 개수(중복), 실제 판매점 개수, 비율을 표시"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
from matplotlib import rc
rc('font', family = 'Arial') #실행환경이 Mac일 경우
np.set_printoptions(legacy='1.13') #np.어쩌구 표시를 안하게 하기
os.chdir(os.path.dirname(__file__))

tai1_count_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/붕어빵_300_2500_all_count_tai.xlsx')
tai2_count_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/스타벅스_300_2500_all_count_tai.xlsx')
# tai1_count_df = pd.read_excel('20250203_dagn_Apr10_최소반경수정100_all_countDistance_tai.xlsx')
# tai2_count_df = pd.read_excel('Starbucks_Apr10_최소반경수정100_all_countDistance_tai.xlsx')
station_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/20250205_서울시내 역(station_df, cent_df 통합).xlsx')

# print(tai1_count_df.head()) #데이터 잘 불러왔는지 확인용


def CC_TAI(station_df, tai_count_df, radius):
    station_labels = station_df['역사명']
    # print(station_labels)
    # tai1_labels = tai1_count_df['역사명']
    df = []
    x = []
    y = []
    for station in station_labels :
        cc = station_df[station_df['역사명'] == station]['Closeness Centrality'].values[0]
        # print(f'{station} : {cc}')
        tai_count = tai_count_df[tai_count_df['역사명'] == station][radius].values[0]
        # print(f'{station} : {tai_count}')
        # x, y  = cc, tai_count
        df.append([x, y])
        x.append(cc)
        y.append(tai_count)
    return x, y ,df
    
# radius = 1500
# tai1_x, tai1_y, tai1_df = CC_TAI(station_df=station_df, tai_count_df=tai1_count_df, radius = radius)
# tai2_x, tai2_y, tai2_df = CC_TAI(station_df=station_df, tai_count_df=tai2_count_df, radius = radius)

# t,p = stats.pearsonr(tai1_x, tai1_y)
# print(f"{radius}m 붕어빵| 피어슨(선형) | t : {t}, p-value: {p}")
# t,p = stats.kendalltau(tai1_x, tai1_y)
# print(f"{radius}m 붕어빵| 켄달(순위)   | t : {t}, p-value: {p}")
# # t,p = stats.spearmanr(tai1_x, tai1_y)
# # print(f"{radius}m 붕어빵|| t : {t}, p-value: {p}")
    
# t,p = stats.pearsonr(tai2_x, tai2_y)
# print(f"{radius}m 스타벅스| 피어슨(선형) | t : {t}, p-value: {p}")
# t,p = stats.kendalltau(tai2_x, tai2_y)
# print(f"{radius}m 스타벅스| 켄달(순위)   | t : {t}, p-value: {p}")
# t,p = stats.spearmanr(tai2_x, tai2_y)
# print(f"{radius}m || t : {t}, p-value: {p}")
# print((stats.spearmanr(tai2_x, tai2_y)))

# def Corre(x_df, y_df, radius):
    # t,p = stats.pearsonr(tai1_x, tai1_y)
    # print(f"{radius}m 붕어빵| 피어슨(선형) | t : {t}, p-value: {p}")
    # t,p = stats.kendalltau(tai1_x, tai1_y)
    # print(f"{radius}m 붕어빵| 켄달(순위)   | t : {t}, p-value: {p}")
    
    # t,p = stats.pearsonr(tai2_x, tai2_y)
    # print(f"{radius}m 스타벅스| 피어슨(선형) | t : {t}, p-value: {p}")
    # t,p = stats.kendalltau(tai2_x, tai2_y)
    # print(f"{radius}m 스타벅스| 켄달(순위)   | t : {t}, p-value: {p}")
# print(df)
# print(sum(tai1_y))
# print(sum(tai2_y))
def Scatter_tai_tai(radius_m, save = False):
    tai1_x, tai1_y, tai1_df = CC_TAI(station_df=station_df, tai_count_df=tai1_count_df, radius = radius_m)
    tai2_x, tai2_y, tai2_df = CC_TAI(station_df=station_df, tai_count_df=tai2_count_df, radius = radius_m)

    #! 폰트 크기(제목과 축 이름은 전부 굵게)
    title_font = {
    'fontsize': 34,
    'fontweight': 1000
    }
    label_font = {
    'fontsize' : 32
    ,'fontweight':1000

    }
    plt.figure(figsize=(12, 9))
    plt.subplot()
    # scatter line
    # plt.scatter(tai1_x, tai1_y, alpha=1,facecolors='none',edgecolors='red')
    # plt.scatter(tai2_x, tai2_y,marker = '*', s=90, alpha=1, facecolors='none',edgecolors='darkgreen')
    # scatter fill
    plt.grid(linestyle='-', linewidth =0.5, alpha = 0.5, which= 'both')
    plt.scatter(tai1_x, tai1_y, alpha=0.8, c = 'red', marker = 'o')
    plt.scatter(tai2_x, tai2_y,marker = '*', s=90, alpha=0.8, c = 'darkgreen')
    plt.title(f'            [{radius_m}]m', fontdict=title_font, pad=20, loc='left')
    # plt.ylim(0,0.002)
    
    plt.xlabel('Closeness Centrality', fontdict=label_font, labelpad= 16)
    plt.ylabel('판매점 개수',font = 'AppleGothic', fontsize = 36, fontweight = 1000, labelpad= 14)
    
    ax = plt.gca()
    plt.legend(['붕어빵', '스타벅스'], loc='upper left',prop={'family':'AppleGothic', 'weight': 'bold','size':20})
    # plt.tight_layout(w_pad=1)
    plt.tick_params(axis = 'y' , which = 'major' , width = 2, length = 5, pad = 3, labelsize = 18)
    plt.tick_params(axis = 'x' , which = 'major' , width = 2, length = 5, pad = 3, labelsize = 18)
    ax.set_yticklabels(ax.get_yticks(), fontsize = 20, fontweight = 'bold')
    ax.set_xticklabels(ax.get_xticks(), fontsize = 20, fontweight = 'bold')
    t,p = stats.pearsonr(tai1_x, tai1_y)
    print(f"{radius_m}m 붕어빵| 피어슨(선형) | r : {t}, p-value: {p}")
    plt.text(0.51, 1.13, f'붕어빵| Pearson| r : {t:.4f}, p-value: {p:.4g}', transform=plt.gca().transAxes, fontweight = 'bold',font = 'AppleGothic',fontsize=13)
    t,p = stats.kendalltau(tai1_x, tai1_y)
    print(f"{radius_m}m 붕어빵| 켄달(순위)   | t : {t}, p-value: {p}")
    plt.text(0.51, 1.09, f'붕어빵| Kendall | t : {t:.4f}, p-value: {p:.4g}', transform=plt.gca().transAxes, fontweight = 'bold',font = 'AppleGothic',fontsize=13)
    
    t,p = stats.pearsonr(tai2_x, tai2_y)
    print(f"{radius_m}m 스타벅스| 피어슨(선형) | r : {t}, p-value: {p}")
    plt.text(0.5, 1.05, f'스타벅스| Pearson| r : {t:.4f}, p-value: {p:.4g}', transform=plt.gca().transAxes,fontweight = 'bold',font = 'AppleGothic', fontsize=13)
    t,p = stats.kendalltau(tai2_x, tai2_y)
    print(f"{radius_m}m 스타벅스| 켄달(순위)   | t : {t}, p-value: {p}")
    plt.text(0.5, 1.01, f'스타벅스| Kendall | t : {t:.4f}, p-value: {p:.4g}', transform=plt.gca().transAxes,fontweight = 'bold',font = 'AppleGothic', fontsize=13)
    
    # #of data, store 텍스트 표시
    plt.text(1.01, .5, f'# of data \n  {sum(tai1_y)}\n  {sum(tai2_y)}', transform=plt.gca().transAxes, fontsize=14,fontweight = 'bold')
    plt.text(1.01, .4, f'# of store \n  {1649}\n  {623}', transform=plt.gca().transAxes, fontsize=14,fontweight = 'bold')
    plt.text(1.01, .3, f'ratio \n  {sum(tai1_y)/1649:.3g}\n  {sum(tai2_y)/623:.3g}', transform=plt.gca().transAxes, fontsize=14,fontweight = 'bold')
    # plt.text(0.8, .5, f'{sum(tai1_y)}', transform=plt.gca().transAxes, fontsize=12)
    
    print(f'# of data|{sum(tai1_y)}')
    print(f'# of data|{1649}')
    print(f'# of data|{sum(tai2_y)}')
    print(f'# of data|{623}')
    if save == True:
        plt.savefig(f'{radius_m}m.png')
    else:
        plt.show()
#todo
# Scatter_tai_tai(radius_m=300)
for column in np.arange(300, 1510, 100):
    Scatter_tai_tai(radius_m=column, save = True)
for column in [2000,2500]:
    Scatter_tai_tai(radius_m=column, save = True)
