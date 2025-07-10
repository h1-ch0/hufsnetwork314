"""1)subplot(1,2,figsize())
1:1비율 jointplot tai vs starbucks 비교 png (1500m)

2)JSD tai vs starbucks 비교 png 
"""

#! #!jointsplineplot을 위한 레이아웃
 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import os
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
rc('font', family = 'AppleGothic') #실행환경이 Mac일 경우

os.chdir(os.path.dirname(__file__))

cent_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/붕어빵_1500_2500_all_count_tai.xlsx')
tai1 = '붕어빵' #!tai에 맞게 설정해주자.. 
tai2 = '스타벅스' #!tai에 맞게 설정해주자.. 
poly1_color = 'skyblue' #!그래프 색도,,,
hist1_color = '#0080FF' #!색상표.ipynb참고하기..
poly2_color = 'green' #!그래프 색도,,,
hist2_color = '#86B404' #!색상표.ipynb참고하기..
# count_df = pd.read_excel(f'{tai}_all_count_tai.xlsx')
count1_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/붕어빵_1500_2500_all_count_tai.xlsx')
# count1_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/20250203_Count.xlsx')
# count2_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/스타벅스_all_count_tai.xlsx')
count2_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/INPUTS_update_0417/스타벅스_1500_2500_all_count_tai.xlsx')

def Scubic_spline(data, data_column, count_df, count_column, set_bins = False):
    x = data[data_column]
    max_x = count_df[count_column].max()
    counts, bins = np.histogram(x, bins=max_x, density=False)
    if set_bins != False:
        counts, bins = np.histogram(x, bins=set_bins, density=False)
    else :
        bins = bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # cs = interpolate.CubicSpline(bin_centers, counts) # 정확한 cubic_spline 계산
    cs = UnivariateSpline(bin_centers, counts) # smoothing 정도를 계산 가능
    x_interp = np.linspace(bin_centers.min(), bin_centers.max(), 300)
    y_interp = cs(x_interp)
    return x_interp, y_interp

def Joint_SplinePlot_FixedCentralityBins(cent_df, count1_df,count2_df, column = 1000, bw_adjust = 0.5,set_bins = False,\
    x_zero = 1, y_zero = 1, save = False, s = 100):    
    x = cent_df["Closeness Centrality"]
    y1 = count1_df[column]
    y2 = count2_df[column]

    #Axes 크기 설정
    left, width = 0.055, 0.3
    bottom, height = 0.1, 0.6
    sub_height = 0.2
    sub_width = 0.1
    spacing = 0.009
    
    rect_main1 = [left, bottom, width, height] ## 산포도, 2kde가 그려질 Main Axes 
    rect_histx1 = [left, bottom + height + spacing, width, sub_height] ## x축 히스토그램이 그려질 Axes
    rect_histy1 = [width + spacing + left, bottom, sub_width, height] ## y축 히스토그램이 그려질 Axes
    rect_main2 = [left+ width + 0.11+ sub_width +spacing, bottom, width, height] ## 산포도, 2kde가 그려질 Main2 Axes 
    rect_histx2 = [left+ width +0.11 + sub_width +spacing, bottom + height + spacing, width, sub_height] ## x2축 히스토그램이 그려질 Axes
    rect_histy2 = [left+ width +0.11 + sub_width +spacing+ width+ spacing, bottom, sub_width, height] ## y2축 히스토그램이 그려질 Axes
    
    fig = plt.figure(figsize=(20, 10)) #3:2비율
    fig.set_facecolor('white')
    
    #! 폰트 크기(제목과 축 이름은 전부 굵게)
    title_font = {
    'fontsize': 20,
    'fontweight': 'bold'
    }
    label_font = {
    'fontsize' : 25
    ,'fontweight': 'bold'
    
    }
    #그리드 설정
    ## 3개 Axes 생성, axes 설정
    ax_main1 = plt.axes(rect_main1)
    ax_main2 = plt.axes(rect_main2)
    # ax_main.set_xlim((0.02, x.max()+0.01))
    # ax_main.set_ylim((0.0, y.max()+4.99))
    ax_main1.tick_params(direction='in', top=True, right=True, labelsize = 15) ## 눈금은 Axes 안쪽으로 설정
    ax_main2.tick_params(direction='in', top=True, right=True, labelsize = 15) ## 눈금은 Axes 안쪽으로 설정
    #! sub1
    ax_histx1 = plt.axes(rect_histx1)
    ax_histx1.tick_params(direction='in', labelbottom=False)
    ax_histy1 = plt.axes(rect_histy1)
    ax_histy1.tick_params(direction='in', labelleft=False)
    ax_main1.grid(True, axis='both', which = 'major', color='gray', linestyle='-', linewidth=0.2 )    
    ax_histx1.grid(True, axis='both', which = 'major', color='gray', linestyle='-', linewidth=0.2 )    
    ax_histy1.grid(True, axis='both', which = 'major', color='gray', linestyle='-', linewidth=0.2 )    
    #! sub2
    ax_histx2 = plt.axes(rect_histx2)
    ax_histx2.tick_params(direction='in', labelbottom=False)
    ax_histy2 = plt.axes(rect_histy2)
    ax_histy2.tick_params(direction='in', labelleft=False)
    ax_main2.grid(True, axis='both', which = 'major', color='gray', linestyle='-', linewidth=0.2 )    
    ax_histx2.grid(True, axis='both', which = 'major', color='gray', linestyle='-', linewidth=0.2 )    
    ax_histy2.grid(True, axis='both', which = 'major', color='gray', linestyle='-', linewidth=0.2 ) 
    
    slope1 = y1.max()/x.max()
    slope2 = y2.max()/x.max()
    x_range = np.arange(0, 1, 0.001)
    y1_range = slope1*x_range
    y2_range = slope2*x_range
    ax_main1.plot(x_range, y1_range, label = '(0,0),(max,max)', color = 'red') ##! 기준선 생성
    ax_main1.set_xlim((0.02, x.max()+0.01))
    ax_main1.set_ylim((0.0, slope1*(x.max()+0.01)))
    ax_main1.yaxis.set_major_locator(MultipleLocator(10))
    ax_main1.yaxis.set_minor_locator(MultipleLocator(1))
    ax_main2.plot(x_range, y2_range, label = '(0,0),(max,max)', color = 'red') ##! 기준선 생성
    ax_main2.set_xlim((0.02, x.max()+0.01))
    ax_main2.set_ylim((0.0, slope2*(x.max()+0.01)))
    ax_main2.yaxis.set_major_locator(MultipleLocator(10))
    ax_main2.yaxis.set_minor_locator(MultipleLocator(1))
    if x_zero == 0: #True일 경우 0부터 표시되도록
        ax_main1.set_xticks(np.arange(0,round(x.max()+0.01,2), 0.005)) 
        ax_main1.xaxis.set_major_locator(MultipleLocator(0.01))
        ax_main1.xaxis.set_minor_locator(MultipleLocator(0.005))
        ax_main2.set_xticks(np.arange(0,round(x.max()+0.01,2), 0.005)) 
        ax_main2.xaxis.set_major_locator(MultipleLocator(0.01))
        ax_main2.xaxis.set_minor_locator(MultipleLocator(0.005))
        
    else:
        ax_main1.set_xticks(np.arange(0.02,round(x.max()+0.01,2), 0.005)) 
        ax_main2.set_xticks(np.arange(0.02,round(x.max()+0.01,2), 0.005)) 

    current_values = ax_main1.get_xticks()
    ax_main1.set_xticklabels(['{:g}'.format(x) for x in current_values]) # .nf로 소수점 자리수 설정 가능 g는 끝0 제거거
    current_values = ax_main2.get_xticks()
    ax_main2.set_xticklabels(['{:g}'.format(x) for x in current_values]) # .nf로 소수점 자리수 설정 가능 g는 끝0 제거거
    
    # 각 Axes에 히스토그램 그리기
    #!sub1
    bins = y1.max()
    x_counts, _ = np.histogram(x, bins=20)
    y_counts, _ = np.histogram(y1, bins=bins)
    histx_x_interp, histx_y1_interp = Scubic_spline(cent_df,'Closeness Centrality',count1_df, column , set_bins = set_bins) #x축 spline 계산(bins는 변수입력, 안 넣으면 count_df따라감)
    ax_histx1.plot(histx_x_interp,histx_y1_interp,  color=poly1_color, lw=1, label='Cubic Spline') #y축 spline 그리기 (centrality)
    ax_histx1.hist(x, bins= 20, density= False,color=hist1_color) ## x축 히스토그램
    histy_x_interp, histy_y1_interp = Scubic_spline(count1_df,column, count1_df, column) #y축 spline 계산
    ax_histy1.plot(histy_y1_interp,histy_x_interp,  color=poly1_color, lw=1, label='Cubic Spline') #y축 spline 그리기#0080FF#FFFF00#86B404
    ax_histy1.hist(y1, bins= bins, orientation='horizontal',color = hist1_color) ## y축 히스토그램

    sns.kdeplot(ax = ax_main1 , bw_adjust= 0.5, fill= True, \
        x = x, y = y1) ## 2D kde 생성
    #!sub2
    bins = y2.max()
    x_counts, _ = np.histogram(x, bins=20)
    y2_counts, _ = np.histogram(y2, bins=bins)
    histx_x_interp, histx_y2_interp = Scubic_spline(cent_df,'Closeness Centrality',count2_df, column , set_bins = set_bins) #x축 spline 계산(bins는 변수입력, 안 넣으면 count_df따라감)
    ax_histx2.plot(histx_x_interp,histx_y2_interp,  color=poly2_color, lw=1, label='Cubic Spline') #y축 spline 그리기 (centrality)
    ax_histx2.hist(x, bins= 20, density= False,color=hist2_color) ## x축 히스토그램
    histy_x_interp, histy_y2_interp = Scubic_spline(count2_df,column, count2_df, column) #y축 spline 계산
    ax_histy2.plot(histy_y2_interp,histy_x_interp,  color=poly2_color, lw=1, label='Cubic Spline') #y축 spline 그리기#0080FF#FFFF00#86B404
    ax_histy2.hist(y2, bins= bins, orientation='horizontal',color = hist2_color) ## y축 히스토그램

    sns.kdeplot(ax = ax_main2 , bw_adjust= 0.5, fill= True, \
        x = x, y = y2)

    #!sub1
    ax_histx1.set_ylim(0, None)
    ax_histx1.set_xlim(ax_main1.get_xlim()) 
    ax_histy1.set_xlim(0, None)
    ax_histy1.set_ylim(ax_main1.get_ylim())
    # plt.suptitle('[CC에 대한 지하철 역의 개수 분포]',x=0.28, y=0.89 , fontsize = 10)
    # plt.suptitle(f'[지하철 역에 대하여 {column}m 이내의 스타벅스 지점 분포]',x=0.28, y=0.89 , fontsize = 10)
    # ax_histx.set_title('[CC에 대한 지하철 역의 개수 분포]')
    # ax_histy.set_title(f'[지하철 역에 대하여 \n{column}m 이내의 {tai} 지점 분포]', fontdict=title_font)
    ax_histy1.set_title(f'[{column}m]', fontdict=title_font)
    ax_main1.set_xlabel('Closeness Centrality', loc = 'center',labelpad = 10 , fontdict = label_font) ## 산포도 x 라벨
    ax_main1.set_ylabel(f'[{tai1}] 판매점 개수' , labelpad = 10 , fontdict = label_font) ## 산포도 y 라벨
    ax_main1.legend(loc='upper left')
    # plt.title(f'[{column}]m ', loc = 'right' , fontdict= title_font)
    #!sub2
    ax_histx2.set_ylim(0, None)
    ax_histx2.set_xlim(ax_main2.get_xlim()) 
    ax_histy2.set_xlim(0, None)
    ax_histy2.set_ylim(ax_main2.get_ylim())
    # plt.suptitle('[CC에 대한 지하철 역의 개수 분포]',x=0.28, y=0.89 , fontsize = 10)
    # plt.suptitle(f'[지하철 역에 대하여 {column}m 이내의 스타벅스 지점 분포]',x=0.28, y=0.89 , fontsize = 10)
    # ax_histx.set_title('[CC에 대한 지하철 역의 개수 분포]')
    # ax_histy.set_title(f'[지하철 역에 대하여 \n{column}m 이내의 {tai} 지점 분포]', fontdict=title_font)
    ax_histy2.set_title(f'[{column}m]', fontdict=title_font)
    ax_main2.set_xlabel('Closeness Centrality', loc = 'center',labelpad = 10 , fontdict = label_font) ## 산포도 x 라벨
    ax_main2.set_ylabel(f'[{tai2}] 지점 개수' , labelpad = 10 , fontdict = label_font) ## 산포도 y 라벨
    ax_main2.legend(loc='upper left')
    
    #저장여부
    if save == True: #저장을 할 지, 플롯을 보여줄 지 여부에 따라 실행. 기본값은 False
        plt.savefig(f'joint_{column}m_kde(bw={bw_adjust})(x=0.00).png')
    else:
        plt.show()
        
        
x = np.arange(1500,2510,100)
for column in x:        
    Joint_SplinePlot_FixedCentralityBins(cent_df, count1_df=count1_df, count2_df = count2_df, column = column, bw_adjust = 0.5,set_bins = 20,\
    x_zero = 0, y_zero = 1, save = True, s = 100)