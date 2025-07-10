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
from matplotlib.ticker import MultipleLocator,FuncFormatter
rc('font', family = 'Arial') #실행환경이 Mac일 경우
np.set_printoptions(legacy='1.13') #np.어쩌구 표시를 안하게 하기
os.chdir(os.path.dirname(__file__))

taiOne_count_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/붕어빵_300_2500_all_count_tai.xlsx')
taiTwo_count_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/스타벅스_300_2500_all_count_tai.xlsx')
cent_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/20250205_서울시내 역(station_df, cent_df 통합).xlsx')

# print(taiOne_count_df.head()) #데이터 잘 불러왔는지 확인용
# print(taiTwo_count_df.head()) #데이터 잘 불러왔는지 확인용
# print(cent_df.head()) #데이터 잘 불러왔는지 확인용


def CC_TAI(station_df, tai_count_df, radius):#x축 CC, y축 판매점 개수
    station_labels = station_df['역사명']
    # print(station_labels)
    # taiOne_labels = taiOne_count_df['역사명']
    xy = []
    x = []
    y = []
    for station in station_labels :
        cc = station_df[station_df['역사명'] == station]['Closeness Centrality'].values[0]
        # print(f'{station} : {cc}')
        tai_count = tai_count_df[tai_count_df['역사명'] == station][radius].values[0]
        # print(f'{station} : {tai_count}')
        # x, y  = cc, tai_count
        xy.append([x, y])
        x.append(cc)
        y.append(tai_count)
    return x, y ,xy
def Pearson_tai(cent_df, tai_df, radius):
    cc_x, tai_y, tai_df = CC_TAI(cent_df, tai_df, radius)
    r, p = stats.pearsonr(cc_x, tai_y)
    return r, p

def Kendall_tai(cent_df, tai_df, radius):
    cc_x, tai_y, tai_df = CC_TAI(cent_df, tai_df, radius)
    t, p = stats.kendalltau(cc_x, tai_y)
    return t, p

def Fig_Pearson_tai_tai(taiOne_df,taiTwo_df, count_columns):
    # plt.figure(figsize=(12, 9))
    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(10)
    
        #! 폰트 크기(제목과 축 이름은 전부 굵게)
    title_font = {
    'fontsize': 34,
    'fontweight': 1000
    }
    label_font = {
    'fontsize' : 32
    ,'fontweight':1000

    }
    # scatter line
    # plt.scatter(taiOne_x, taiOne_y, alpha=1,facecolors='none',edgecolors='red')
    # plt.scatter(taiTwo_x, taiTwo_y,marker = '*', s=90, alpha=1, facecolors='none',edgecolors='darkgreen')
    # scatter fill
    # plt.grid(linestyle='-', linewidth =0.5, alpha = 0.5, which= 'both')
    # plt.title(f'            [{radius_m}]m', fontdict=title_font, pad=20, loc='left')
    # plt.ylim(0,0.002)
    
    plt.xlabel('Radius(m)', fontdict=label_font, labelpad= 16)
    plt.ylabel('Pearson correlation coefficient', fontsize = 36, fontweight = 1000, labelpad= 14)
    
    
    # ax = plt.gca()
    # plt.tight_layout(w_pad=1)
    plt.xticks(np.arange(0, 1600, 100))
    ax.set_yticklabels(ax.get_yticks(), fontsize = 20, fontweight = 'bold')
    ax.set_xticklabels(ax.get_xticks(), fontsize = 20, fontweight = 'bold')
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.tick_params(axis = 'y' , which = 'major' , width = 2, length = 8, pad = 3, labelsize = 20)
    ax.tick_params(axis = 'y' , which = 'minor' , width = 1, length = 5, pad = 3, labelsize = 20)
    ax.tick_params(axis = 'x' , which = 'major' , width = 2, length = 6, pad = 3, labelsize = 22)
    ax.tick_params(axis = 'x', which = 'minor' , width = 1.5, length = 5, direction = 'out' , labelsize = 20, pad = 10)
    
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}'))  # 실제 값 표시
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))  # 실제 값 표시
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # x축 정수로 표시
    
    for column in count_columns:
        taiOne_pr, taiOne_pp = Pearson_tai(cent_df=cent_df, tai_df=taiOne_df, radius = column)
        print(f"붕어빵| 피어슨(선형) | r : {taiOne_pr}, p-value: {taiOne_pp}")
        taiTwo_pr, taiTwo_pp = Pearson_tai(cent_df=cent_df, tai_df=taiTwo_df, radius = column)
        print(f"스타벅스| 피어슨(선형) | r : {taiTwo_pr}, p-value: {taiTwo_pp}")
        plt.scatter(column, taiTwo_pr,marker = '*', s= 240, alpha=1-taiTwo_pp, c = 'darkgreen')
        plt.scatter(column, taiOne_pr,marker = 'o', s= 130, alpha=1-taiOne_pp,c = 'red')
        # todo 이제 대충 보여주고, 제대로 나오는지 확인하면 레이아웃 수정하자..
    leg = plt.legend(['스타벅스','붕어빵'], loc='upper left',prop={'family':'AppleGothic', 'weight': 'bold','size':20})

    for handle in leg.legend_handles:
        handle.set_alpha(1)  # legend 마커만 alpha 조절
    # plt.legend(['스타벅스','붕어빵'], loc='upper left',prop={'family':'AppleGothic', 'weight': 'bold','size':20})
    ax.set_xlim(200,1600)
    ax.set_xlim(1300,2600)
    ax.set_ylim(0,0.7)
    plt.show()

def Fig_Kendalltau_tai_tai(taiOne_df,taiTwo_df, count_columns):
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(10)
    
        #! 폰트 크기(제목과 축 이름은 전부 굵게)
    title_font = {
    'fontsize': 34,
    'fontweight': 1000
    }
    label_font = {
    'fontsize' : 32
    ,'fontweight':1000

    }
    
    plt.xlabel('Radius(m)', fontdict=label_font, labelpad= 16)
    plt.ylabel('Kendall correlation coefficient', fontsize = 36, fontweight = 1000, labelpad= 14)
    
    
    # plt.tight_layout(w_pad=1)
    plt.xticks(np.arange(0, 1600, 100))
    ax.set_yticklabels(ax.get_yticks(), fontsize = 20, fontweight = 'bold')
    ax.set_xticklabels(ax.get_xticks(), fontsize = 20, fontweight = 'bold')
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.tick_params(axis = 'y' , which = 'major' , width = 2, length = 8, pad = 3, labelsize = 20)
    ax.tick_params(axis = 'y' , which = 'minor' , width = 1, length = 5, pad = 3, labelsize = 20)
    ax.tick_params(axis = 'x' , which = 'major' , width = 2, length = 6, pad = 3, labelsize = 22)
    ax.tick_params(axis = 'x', which = 'minor' , width = 1.5, length = 5, direction = 'out' , labelsize = 20, pad = 10)
    
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}'))  # 실제 값 표시
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))  # 실제 값 표시
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # x축 정수로 표시
    
    for column in count_columns:
        taiOne_kt, taiOne_kp = Kendall_tai(cent_df=cent_df, tai_df=taiOne_df, radius = column)
        print(f"붕어빵| 켄달타우(순위) | r : {taiOne_kt}, p-value: {taiOne_kp}")
        taiTwo_kt, taiTwo_kp = Kendall_tai(cent_df=cent_df, tai_df=taiTwo_df, radius = column)
        print(f"스타벅스| 켄달타우(순위) | r : {taiTwo_kt}, p-value: {taiTwo_kp}")
        plt.scatter(column, taiTwo_kt,marker = '*', s=240, alpha=0.8, c = 'darkgreen')
        plt.scatter(column, taiOne_kt, alpha=0.8, s = 130,c = 'red', marker = 'o')
        # todo 이제 대충 보여주고, 제대로 나오는지 확인하면 레이아웃 수정하자..
    plt.legend(['스타벅스','붕어빵'], loc='upper left',prop={'family':'AppleGothic', 'weight': 'bold','size':20})
    # ax.set_xlim(200,1600)
    ax.set_xlim(1300,2600)
    ax.set_ylim(0,0.5)
    plt.show()

#todo==========
# Fig_Pearson_tai_tai(taiOne_df=taiOne_count_df,taiTwo_df=taiTwo_count_df,count_columns=np.arange(300,1510,100))#테스트용
# Fig_Kendalltau_tai_tai(taiOne_df=taiOne_count_df,taiTwo_df=taiTwo_count_df,count_columns=np.arange(300,1510,100))#테스트용
Fig_Pearson_tai_tai(taiOne_df=taiOne_count_df,taiTwo_df=taiTwo_count_df,count_columns=[2000,2500])#테스트용
# Fig_Kendalltau_tai_tai(taiOne_df=taiOne_count_df,taiTwo_df=taiTwo_count_df,count_columns=[2000,2500])#테스트용
