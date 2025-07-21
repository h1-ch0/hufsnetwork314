import pandas as pd
import networkx as nx
import folium
import matplotlib.pyplot as plt
import os,glob,re

# def export_edges_to_excel(G, filename="edges.xlsx"):
#     edge_data = []
#     for u, v, data in G.edges(data=True):
#         edge_data.append({
#             'from': G.nodes[u].get('name', str(u)),
#             'to': G.nodes[v].get('name', str(v)),
#             'lines': ', '.join(data.get('lines', [])),
#             'colors': ', '.join(data.get('colors', [])),
#             'from_longitude': u[0],
#             'from_latitude': u[1],
#             'to_longitude': v[0],
#             'to_latitude': v[1],
#         })
#     df = pd.DataFrame(edge_data)
#     df.to_excel(filename, index=False)
#     print(f"엣지 정보가 {filename}에 저장되었습니다.")

def get_unique_filename(base_path, pattern):
    """
    base_path: 파일이 저장될 경로(ex: './data')
    pattern: 파일명 패턴(ex: 'sample.txt' 또는 'report_*.csv')

    반환값: 겹치지 않는 파일 경로 문자열
    """
    # 파일 전체 경로 생성
    path_pattern = os.path.join(base_path, pattern)

    # 패턴에 맞는 모든 파일 찾기
    matched_files = glob.glob(path_pattern)
    if not matched_files:
        # 같거나 유사한 파일이 아예 없다면 그냥 pattern 이름 반환
        return os.path.join(base_path, pattern)

    # pattern에서 확장자 분리
    basename, ext = os.path.splitext(pattern)

    # 이미 (N) 형식으로 붙은 게 있는지 체크
    regex = re.compile(rf"^{re.escape(basename)}(?: \((\d+)\))?{re.escape(ext)}$")
    numbers = []
    for file in matched_files:
        fname = os.path.basename(file)
        match = regex.match(fname)
        if match:
            num = match.group(1)
            numbers.append(int(num) if num is not None else 0)
            
    max_n = max(numbers) if numbers else 0
    new_n = max_n + 1

    # 새 파일명 생성
    if new_n == 1:
        new_filename = f"{basename} (1){ext}"
    else:
        new_filename = f"{basename} ({new_n}){ext}"
    return os.path.join(base_path, new_filename)

# 사용 예시
# file_path = get_unique_filename('filepath', 'report_*.xlsx')




def import_from_excel(filename):
    """엑셀 파일에서 노드 정보를 읽어 NetworkX 그래프 재구축"""
    # 엑셀 파일 읽기
    # file_path = get_unique_filename('/', filename)

    df = pd.read_excel(filename)
    
    # 그래프 생성
    G = nx.Graph()
    
    # 노드 추가
    for _, row in df.iterrows():
        node_id = row['node_id']
        # 좌표 정보 복원
        pos = (float(row['longitude']), float(row['latitude']))
        
        # 노드 속성 설정
        attributes = {
            'name': row['name'],
            'aliases': set(row['aliases'].split(', ')) if pd.notna(row['aliases']) else set(),
            'lines': set(row['lines'].split(', ')) if pd.notna(row['lines']) else set(),
            'color': row['color'],
            'pos': pos
        }
        G.add_node(node_id, **attributes)
    
    # 엣지 추가 (이전/다음 역 정보 사용)
    for _, row in df.iterrows():
        node_id = row['node_id']
        
        # 이전 역 연결
        if pd.notna(row['prev_stations']):
            for prev_node in row['prev_stations'].split(', '):
                if prev_node in G.nodes:
                    # print(attributes)
                    G.add_edge(node_id, prev_node, **attributes)
        
        # 다음 역 연결
        if pd.notna(row['next_stations']):
            for next_node in row['next_stations'].split(', '):
                if next_node in G.nodes:
                    G.add_edge(node_id, next_node)
    
    return G
def create_folium_map_enhanced(G, zoom_start=12):
    positions = nx.get_node_attributes(G, 'pos')
    latitudes = [pos[0] for pos in positions.values()]
    longitudes = [pos[1] for pos in positions.values()]
    left, right, bottom, top = min(longitudes), max(longitudes), min(latitudes), max(latitudes)
    m = folium.Map(location=[(bottom+top)/2, (left+right)/2],
                   tiles='cartodbpositron',
                   zoom_start=zoom_start,
                   width=1800, height=1500,
                   control_scale=True)
    # 노드 추가
    for node, data in G.nodes(data=True):
        line_list = ', '.join(sorted(data['lines']))
        alias_list = ', '.join(sorted(data['aliases'] - {data['name']}))
        popup_html = f"<b>{data['name']}</b>"
        if alias_list:
            popup_html += f" ({alias_list})"
        popup_html += f"<br><b>노선:</b> {line_list}"
        # print(data['pos'])
        folium.CircleMarker(
            location=data['pos'],  # lat, lon 순서로 위치 지정
            radius=5, color=data['color'], fill=True,
            fill_color=data['color'], fill_opacity=1,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

    for u, v, edata in G.edges(data=True):
        # print(f"Adding edge from u: {u}\b to u: {v} \bwith data: {edata}")  # 엣지 정보 확인용
        # 여러 노선이 겹치는 경우, 각 노선별로 PolyLine을 그림
        for line, color in zip(edata['lines'], edata['color']):
            # print(edata['color'])
            # print(color)
            # 이전역, 다음역, 노선명 popup
            u_name = G.nodes[u]['name']
            v_name = G.nodes[v]['name']
            popup_html = f"<b>{u_name} - {v_name}</b><br><b>노선:</b> {line}"
            location_data = list((list(G.nodes[u]['pos']), list(G.nodes[v]['pos'])))
            print('=============')
            print(location_data)
            #! fucking polyline doesn't work and perplexity, chatGPT don't know why either. 
            folium.PolyLine(
                locations=location_data,
                color=edata['color'], weight=4, opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

    # folium.PolyLine(
    #     locations=location_data,
    #     color=color, weight=4, opacity=0.7,
    #     popup=folium.Popup(popup_html, max_width=300)
    # ).add_to(m)
    return m




def visualize_graph(G, title):
    """
    Visualizes the graph with nodes and edges.
    Parameters:
    G (nx.Graph): A NetworkX graph object representing the routes and stop nodes.
    Returns:
    None: Displays a plot of the graph.
    """
    # Set the plot size
    plt.figure(figsize=(15, 8))  # Adjust the width and height as needed

    # Draw the graph with nodes and edges
    pos = nx.get_node_attributes(G, 'pos')
    node_labels = nx.get_node_attributes(G, 'name')
    node_colours = list(nx.get_node_attributes(G, 'colour').values())
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=100, node_color=node_colours, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)
    plt.title(title)
    plt.show()
    
# 사용 예시
area_search = "Madrid"
date = "15Jul21"
# filename = "/Volumes/YAHO_/009.Git_Repo/hufsnetwork314/Moscow_subway_nodes_Jul21_v2_1.xlsx"
filename = f"{area_search}_subway_nodes_{date}.xlsx"
rebuilt_graph = import_from_excel(filename)
G = rebuilt_graph
# print(G.nodes(data=True))  # 노드 정보 확인
# 그래프 시각화
# visualize_graph(rebuilt_graph, f"Rebuilt {area_search} Subway Network")

# Folium 지도 생성
rebuilt_map = create_folium_map_enhanced(rebuilt_graph)
rebuilt_map.save(f"rebuilt_{area_search}_subway_map_{date}.html")
list_pos = list(nx.get_node_attributes(rebuilt_graph, 'pos').values())
# print(list_pos)  # 첫 5개 노드 위치 확인
# for node in G.nodes(data=True):
#     print(node)  # 노드 정보 확인
# for edges in G.edges:
    # print(edges[0])  # 엣지 정보 확인
    # print(edges[1])  # 엣지 정보 확인