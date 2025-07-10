
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import MultipleLocator, IndexLocator, FuncFormatter, LogLocator, FixedLocator
np.set_printoptions(legacy='1.21') #np.어쩌구 표시를 안하게 하기

# rc('font', family = 'AppleGothic') #실행환경이 Mac일 경우
rc('font', family = 'Arial')

# rc('font', family = 'NanumGothic') #실행환경이 Windows일 경우

os.chdir(os.path.dirname(__file__))
count_columns = [300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500] #모든 거리에 대한 for문을 하기 편하게 하기 위한 설정
# radius_km_list = np.arange(0.3, 1.6, 0.1)
# print(os.listdir("../Inputs"))
'''일단 히스토그램을 그리는 것부터 시작하자.
어떤 데이터든지, 일단 히스토그램을 여러개 겹쳐 그리는 것을...'''
cent_df = pd.read_excel('/Volumes/YAHO_/000. Processing Python/붕어빵/Inputs/20250205_서울시내 역(station_df, cent_df 통합).xlsx')
# JSD_DF  = pd.read_excel(Outputs + 'JSD(Cent_df) station~random_all.xlsx')
tai_mean = pd.read_excel('/Volumes/YAHO_/006.일일 보관/0328/JSD수정본 대표값_JSD 10~1510m tai_mean_to_excel.xlsx')
starbucks_mean = pd.read_excel('/Volumes/YAHO_/006.일일 보관/0328/대표값_스타벅스.xlsx')
tai_star_JSD = pd.read_excel('/Volumes/YAHO_/006.일일 보관/0415/250415_JSD_스타벅스_붕어빵.xlsx')
tai_star_JSD_T = pd.read_excel('/Volumes/YAHO_/006.일일 보관/0416/250416_JSD_스타벅스_붕어빵(붕어빵).xlsx')
tai_star_JSD_S = pd.read_excel('/Volumes/YAHO_/006.일일 보관/0416/250416_JSD_스타벅스_붕어빵(스타벅스).xlsx')

tai1_count_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/0415/붕어빵_300_2500_all_count_tai.xlsx')
tai2_count_df = pd.read_excel('/Volumes/YAHO_/006.일일 보관/0415/스타벅스_300_2500_all_count_tai.xlsx')
# tai_count = pd.read_excel('20250203_Count.xlsx')
# starbucks_count = pd.read_excel('스타벅스_all_count_tai.xlsx')


'''normalize확인해서 excelout만 해결하고 그래프 그리기'''
def calcuate_JSD(count_df_P, count_df_Q,column): #cent_df말고 count count끼리의 JSD를 계산하는 함수
    #!기준을 PQ중 큰 쪽으로
    column_dataP = count_df_P[column].to_list()
    column_dataQ = count_df_Q[column].to_list()
    # plt.figure(figsize = (12,6))
    # print(column_dataP)
    # print(column_dataQ)
    # print(f'++============{column_dataP+column_dataQ}')
    max_value = max(column_dataP+column_dataQ)
    # print(column_dataP+column_dataQ)
    # print(f'max_value : {max_value}')
    # max_value = 50
    countsp, _ = np.histogram(column_dataP, bins=max_value)
    countsq, _ = np.histogram(column_dataQ, bins=max_value)
    prob_p_i = countsp / countsp.sum()
    prob_q_i = countsq / countsq.sum()#np.array
    # prob_q_i = [countsq[i]/sum(countsq) for i in range(len(countsq))]#list
    # print(f'prob_p_i : {type(prob_p_i)}')
    # print(f'prob_q_i : {type(prob_q_i)}')
    # print(prob_p_i.sum())
    # print(prob_q_i.sum())
    P = np.asarray(prob_p_i)
    Q = np.asarray(prob_q_i)
    M = []
    for i in range(len(Q)):
        M.append((Q[i]+P[i])/2)
        # print(M)
    M = np.asarray(M)
    # print(M)
    p_d_js = 1/2*(np.nansum(P*(np.log(P)-np.log(M))))
    q_d_js = 1/2*(np.nansum(Q*(np.log(Q)-np.log(M))))
    JSD = p_d_js+q_d_js
    print(f'{column}m JSD = {p_d_js+q_d_js}')
    return JSD
def calcuate_JSD_PQ(count_df_P, count_df_Q,column): #cent_df말고 count count끼리의 JSD를 계산하는 함수
    #!P기준으로 bins 잡기
    column_dataP = count_df_P[column].to_list()
    column_dataQ = count_df_Q[column].to_list()
    # plt.figure(figsize = (12,6))
    # print(column_dataP)
    # print(column_dataQ)
    # print(f'++============{column_dataP+column_dataQ}')
    max_value = max(column_dataP)
    # print(column_dataP+column_dataQ)
    print(f'max_value : {max_value}')
    # max_value = 50
    countsp, _ = np.histogram(column_dataP, bins=max_value)
    countsq, _ = np.histogram(column_dataQ, bins=max_value)
    prob_p_i = countsp / countsp.sum()
    prob_q_i = countsq / countsq.sum()#np.array
    # prob_q_i = [countsq[i]/sum(countsq) for i in range(len(countsq))]#list
    # print(f'prob_p_i : {type(prob_p_i)}')
    # print(f'prob_q_i : {type(prob_q_i)}')
    # print(prob_p_i.sum())
    # print(prob_q_i.sum())
    P = np.asarray(prob_p_i)
    Q = np.asarray(prob_q_i)
    M = []
    for i in range(len(Q)):
        M.append((Q[i]+P[i])/2)
        # print(M)
    M = np.asarray(M)
    # print(M)
    p_d_js = 1/2*(np.nansum(P*(np.log(P)-np.log(M))))
    q_d_js = 1/2*(np.nansum(Q*(np.log(Q)-np.log(M))))
    JSD = p_d_js+q_d_js
    print(f'{column}m JSD = {p_d_js+q_d_js}')
    return JSD    

def calcuate_JSD_CC(count_df_P, cent_df,column): 
    column_data = count_df_P[column]
    column_dataq = cent_df['Closeness Centrality']
    plt.figure(figsize = (12,6))
    max_value = column_data.max()+1
    counts, _ = np.histogram(column_data, bins=max_value)
    countsq, _ = np.histogram(column_dataq, bins=max_value)
    prob_p_i = counts / counts.sum()
    prob_q_i = [countsq[i]/sum(countsq) for i in range(len(countsq))]
    P = np.asarray(prob_p_i)
    Q = np.asarray(prob_q_i)
    M = []
    for i in range(len(Q)):
        M.append((Q[i]+P[i])/2)
        # print(M)
    M = np.asarray(M)
    # print(M)
    p_d_js = 1/2*(np.nansum(P*(np.log(P)-np.log(M))))
    q_d_js = 1/2*(np.nansum(Q*(np.log(Q)-np.log(M))))
    JSD = p_d_js+q_d_js
    print(f'{column}m JSD = {p_d_js+q_d_js}')
    return JSD
# calcuate_JSD(tai_count, starbucks_count, 1000) #1000m에 대한 JSD를 계산해보자
# calcuate_JSD(starbucks_count,tai_count, 1000) #1000m에 대한 JSD를 계산해보자

# JSD_dict = {}
# for column in count_columns:
#     JSD = calcuate_JSD_PQ(tai1_count_df, tai2_count_df, column) 
#     # calcuate_JSD(starbucks_count, tai_count, column)
#     JSD_dict[column] = JSD
# pd.DataFrame(JSD_dict.items(), columns=['radius_m', 'JSD']).to_excel('250416_JSD_스타벅스_붕어빵(붕어빵).xlsx', index=False)
# JSD_dict = {}
# for column in count_columns:
#     JSD = calcuate_JSD_PQ(tai2_count_df, tai1_count_df, column) 
#     # calcuate_JSD(starbucks_count, tai_count, column)
#     JSD_dict[column] = JSD
# pd.DataFrame(JSD_dict.items(), columns=['radius_m', 'JSD']).to_excel('250416_JSD_스타벅스_붕어빵(스타벅스).xlsx', index=False)


# JSD_dict = {}
# for column in count_columns:
#     JSD = calcuate_JSD_CC(tai1_count_df, cent_df, column) 
#     # calcuate_JSD(starbucks_count, tai_count, column)
#     JSD_dict[column] = JSD
# pd.DataFrame(JSD_dict.items(), columns=['radius_m', 'JSD']).to_excel('250417_JSD_CC붕어빵.xlsx', index=False)
# JSD_dict = {}
# for column in count_columns:
#     JSD = calcuate_JSD_CC(tai2_count_df,cent_df, column) 
#     # calcuate_JSD(starbucks_count, tai_count, column)
#     JSD_dict[column] = JSD
# pd.DataFrame(JSD_dict.items(), columns=['radius_m', 'JSD']).to_excel('250417_JSD_CC스타벅스.xlsx', index=False)
    
start =300
end = 1510
count_columns = np.arange(start,end,100)

fig, ax = plt.subplots()
fig.set_size_inches(15,10) # 3:2 비율
#! 폰트 크기(제목과 축 이름은 전부 굵게)
title_font = {
'fontsize': 36,
'fontweight': 'black'
}
label_font = {
'fontsize' : 34
,'fontweight':'black'

}
fig.set_figwidth(15)
fig.set_figheight(10)
# plt.figure(figsize=(16,10))
# ax.grid(linestyle='-', linewidth =0.5, alpha = 0.5, which= 'both')
JSD_tai = []
for i in count_columns:
    jsd = tai_mean.loc[tai_mean['radius_m'] == int(i), 'JSD'].to_list()[0]
    JSD_tai.append(jsd)
# print(JSD_tai)
JSD_starbucks = []
for i in count_columns:
    jsd = starbucks_mean.loc[starbucks_mean['radius_m'] == int(i), 'JSD'].to_list()[0]
    JSD_starbucks.append(jsd)
JSD_tai_star_T = []
count_columns_2 = [300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000,2500] #모든 거리에 대한 for문을 하기 편하게 하기 위한 설정

for i in count_columns:
    jsd = tai_star_JSD_T.loc[tai_star_JSD_T['radius_m'] == int(i), 'JSD'].to_list()[0]
    JSD_tai_star_T.append(jsd)
    
JSD_tai_star_S = []
for i in count_columns:
    jsd = tai_star_JSD_S.loc[tai_star_JSD_S['radius_m'] == int(i), 'JSD'].to_list()[0]
    JSD_tai_star_S.append(jsd)
tai_star_JSD = tai_star_JSD['JSD'].to_list()
# JSD_starbucks = starbucks_mean["JSD"].to_list()
print(JSD_tai_star_T)
print(JSD_tai_star_S)
print(tai_star_JSD)
# plt.plot(count_columns, tai_star_JSD,marker = 'o', c = 'gray', label = '붕어빵과 스타벅스의 JSD', linewidth = 6 ,markersize=10)
# plt.plot(count_columns, JSD_tai_star_T,marker = '^', c = '#DAA520', label = '붕어빵과 스타벅스의 JSD(붕어빵)', linewidth = 2 ,markersize=10)
# plt.scatter([2000,2500],[0.3143,0.365991], marker= '^', c = '#DAA520', s = 100)

# plt.plot(count_columns, JSD_tai_star_S,marker = '^', c = 'blue', label = '붕어빵과 스타벅스의 JSD(스타벅스)', linewidth = 2 ,markersize=5)
# plt.scatter([2000,2500],[0.285415,0.33149], marker= '^', c = 'blue', s = 100)

plt.plot(count_columns,JSD_starbucks,marker = 'D', c = '#008000', label = 'Starbucks-Closeness centrality', linewidth = 5 ,markersize=15)
plt.scatter([2000,2500],[0.347106927,0.313158134], marker= 'D', c = '#008000', s = 150)
plt.plot(count_columns, JSD_tai,marker = 'o', c = 'red', label = '붕어빵-Closeness centrality', linewidth = 5 ,markersize=17)
plt.scatter([2000,2500],[0.098628801,0.107386729], marker= 'o', c = 'red', s = 170)


ax.xaxis.set_major_locator(MultipleLocator(500))
ax.xaxis.set_minor_locator(MultipleLocator(100))
# ax.xaxis.set_major_formatter(FuncFormatter(plt_tick_log))
ax.tick_params(axis = 'x', which = 'minor' , width = 1.5, length = 5, direction = 'out' , labelsize = 20, pad = 10)
ax.tick_params(axis = 'x', which = 'major' , width = 2, length = 5, direction = 'inout' , labelsize = 20, pad = 10)
ax.set_yticklabels(ax.get_yticks(), fontsize = 20, fontweight = 'bold')
ax.set_xticklabels(ax.get_xticks(), fontsize = 20, fontweight = 'bold')
ax.tick_params(axis = 'y' , which = 'major' , width = 2, length = 5, pad = 3, labelsize = 20)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}'))  # 실제 값 표시
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))  # 실제 값 표시
#? minor 눈금은 100미터 단위 표시
# ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))  # 실제 값 표시
# ax.yaxis.set_minor_locator(MultipleLocator(0.05))
# tick_locations = []
# ax.tick_params(axis = 'y' , which = 'minor' , width = 0.5, length = 3, labelsize = 14)


    
# plt.get_figlabels
plt.yticks(np.arange(0.00, 0.55, 0.1 ))
plt.xticks(np.arange(0, 2510, 500 ))
# plt.xticks(np.arange(start,end,50))
# ax.set_xticklabels(['{:g}m'.format(x) for x in np.arange(start,end,50)])
# plt.title('붕어빵 판매점과 스타벅스 지점 분포의 JSD 비교', fontdict= title_font, pad = 20)
plt.ylabel('Jensen-Shanon Divergence', fontdict=label_font, labelpad= 20)
# plt.xlabel('Radius (m)', fontdict=label_font, labelpad= 14)
ax.legend(loc='lower left',prop={'family':'AppleGothic', 'weight': 'bold','size':14})
# ax.legend(loc='lower left',prop={'family':'AppleGothic', 'weight': 'bold','size':14})
ax.set_xlim(1300,2600)#2000,2500포인트
# ax.set_xlim(200,1550)#300~1500포인트
plt.show()

