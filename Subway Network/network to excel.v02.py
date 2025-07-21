
import matplotlib.pyplot as plt
import requests
import networkx as nx
import folium
from collections import defaultdict
import pandas as pd
date = 'Jul21_v2_1'


def extract_osm_geodata(query):
    overpass_url = "https://lz4.overpass-api.de/api/interpreter"
    params = {'data': query}
    response = requests.get(overpass_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def extract_route_elements(osm_data):
    return [e for e in osm_data['elements'] if 'tags' in e and 'route' in e['tags']]

def extract_node_elements(osm_data):
    return {e['id']: e for e in osm_data['elements'] if e['type'] == 'node'}

def normalize_station_name(tags):
    # 한글/영문/기타 이름 통합
    name = tags.get('name:en') or tags.get('name') or tags.get('name:ko') or None
    aliases = set()
    for k in ['name', 'name:en', 'name:ko']:
        if tags.get(k):
            aliases.add(tags[k])
    return name, aliases

def create_route_graph_integrated(route_elements, node_elements):
    G = nx.Graph()
    # 역 통합: 좌표(소수점 5자리) 기준으로 역을 하나로 묶음
    coord_to_station = dict()
    station_info = dict()  # {coord: {'name':..., 'aliases':..., 'lines':set(), 'pos':(lon,lat)}}
    for route in route_elements:
        line_name = route['tags'].get('name:en') or route['tags'].get('name') or 'UnknownLine'
        # print(line_name)
        line_color = route['tags'].get('colour', '#808080')
        stop_nodes = [m for m in route['members'] if 'stop' in m['role']]
        prev_station = None
        # print(f"Processing route: {line_name} with {len(stop_nodes)} stops")
        for member in stop_nodes:
            ref = member['ref']
            if ref not in node_elements:
                continue
            node = node_elements[ref]
            lon, lat = round(node['lon'], 8), round(node['lat'], 8)
            coord = (lat, lon)
            name, aliases = normalize_station_name(node['tags'])
            # 역 통합
            if name not in station_info:
                station_info[name] = {
                    'name': name,
                    'aliases': set(aliases),
                    'lines': set(),
                    'pos': coord
                }
            else:
                station_info[name]['aliases'].update(aliases)
            station_info[name]['lines'].add(line_name)
            # print(line_name)
            # print(station_info[name]['lines'])
            
            # 그래프에 노드 추가 (중복방지)
            if not G.has_node(name):
                G.add_node(name, pos=coord)
            # 노드 속성 갱신
            G.nodes[name]['name'] = station_info[name]['name']
            G.nodes[name]['aliases'] = station_info[name]['aliases']
            G.nodes[name]['lines'] = station_info[name]['lines']
            

            # 엣지 추가
            if prev_station is not None:
                # 엣지에 노선명/색상 등 추가
                if not G.has_edge(prev_station, name):
                    G.add_edge(prev_station, name, lines=[line_name], colors=[line_color])
                else:
                    if 'lines' not in G.edges[prev_station, name]:
                        G.edges[prev_station, name]['lines'] = []
                    if 'colors' not in G.edges[prev_station, name]:
                        G.edges[prev_station, name]['colors'] = []
                    # 이미 있는 엣지면 노선 추가
                    G.edges[prev_station, name]['lines'].append(line_name)
                    G.edges[prev_station, name]['colors'].append(line_color)
            prev_station = name
    
    
    # 환승역 처리: 2개 이상 노선 포함시 색상 black
    for n, data in G.nodes(data=True):
        # print(data)
        if len(data['lines']) > 2:
            G.nodes[n]['color'] = 'black'
        else:
            # 단일 노선이면 그 노선의 색상
            line = list(data['lines'])[0]
            # 엣지 중 아무거나에서 색상 추출
            color = None
            for nb in G.neighbors(n):
                for l, c in zip(G.edges[n, nb]['lines'], G.edges[n, nb]['colors']):
                    if l == line:
                        color = c
                        break
                if color:
                    break
            G.nodes[n]['color'] = color or '#808080'
    return G

def create_folium_map_enhanced(G, zoom_start=12):
    positions = nx.get_node_attributes(G, 'pos')
    latitudes = [pos[0] for pos in positions.values()]
    longitudes = [pos[1] for pos in positions.values()]
    left, right, bottom, top = min(longitudes), max(longitudes), min(latitudes), max(latitudes)
    m = folium.Map(location=[(bottom+top)/2, (left+right)/2],
                #    tiles='cartodbpositron',
                   tiles = 'OpenStreetMap',
                   zoom_start=zoom_start,
                   width=1800, height=1500,
                   control_scale=True)

    # 엣지 추가
    for u, v, edata in G.edges(data=True):
        # 여러 노선이 겹치는 경우, 각 노선별로 PolyLine을 그림
        # print(u)
        # print(v)
        # print(edata)
        for line, color in zip(edata['lines'], edata['colors']):
            # 이전역, 다음역, 노선명 popup
            u_coord = G.nodes[u]['pos']
            v_coord = G.nodes[v]['pos']
            popup_html = f"<b>{u} - {v}</b><br><b>노선:</b> {line}"
            folium.PolyLine(
                locations=[u_coord,v_coord],
                color=color, weight=4, opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
    # 노드 추가
    for node, data in G.nodes(data=True):
        line_list = ', '.join(sorted(data['lines']))
        alias_list = ', '.join(sorted(data['aliases'] - {data['name']}))
        popup_html = f"<b>{data['name']}</b>"
        if alias_list:
            popup_html += f" ({alias_list})"
        popup_html += f"<br><b>노선:</b> {line_list}"
        folium.CircleMarker(
            location=data['pos'],
            radius=5, color=data['color'], fill=True,
            fill_color=data['color'], fill_opacity=1,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)
    return m
def export_nodes_to_excel(G, filename="nodes.xlsx"):
    node_data = []
    for node, data in G.nodes(data=True):
        # 이전 역(들)과 다음 역(들) 추출
        neighbors = list(G.neighbors(node))
        prev_stations = []
        next_stations = []
        for nb in neighbors:
            # 방향성 없는 그래프라 임의로 분리, 실제로는 순서가 없음
            if node < nb:
                next_stations.append(G.nodes[nb].get('name', str(nb)))
            else:
                prev_stations.append(G.nodes[nb].get('name', str(nb)))
        node_data.append({
            'node_id': str(node),
            'name': data.get('name', ''),
            'aliases': ', '.join(sorted(data.get('aliases', []))),
            'lines': ', '.join(sorted(data.get('lines', []))),
            'color': data.get('color', ''),
            'longitude': G.nodes[node].get('pos')[0],
            'latitude': G.nodes[node].get('pos')[1],
            # 'latitude': node[1],
            'prev_stations': ', '.join(prev_stations),
            'next_stations': ', '.join(next_stations)
        })
    df = pd.DataFrame(node_data)
    df.to_excel(filename, index=False)
    print(f"노드 정보가 {filename}에 저장되었습니다.")

def export_edges_to_excel(G, filename="edges.xlsx"):
    edge_data = []
    for u, v, data in G.edges(data=True):
        edge_data.append({
            'from': G.nodes[u].get('name', str(u)),
            'to': G.nodes[v].get('name', str(v)),
            'lines': ', '.join(data.get('lines', [])),
            'colors': ', '.join(data.get('colors', [])),
            'from_longitude': G.nodes[u].get('pos')[0],
            'from_latitude': G.nodes[u].get('pos')[1],
            'to_longitude': G.nodes[v].get('pos')[0],
            'to_latitude': G.nodes[v].get('pos')[1],
        })
    df = pd.DataFrame(edge_data)
    df.to_excel(filename, index=False)
    print(f"엣지 정보가 {filename}에 저장되었습니다.")

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
    node_colours = list(nx.get_node_attributes(G, 'color').values())
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=100, node_color=node_colours, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)
    plt.title(title)
    plt.show()
        

# ---- 실행 예시 ----
area_search = "Madrid"
# route = subway // 방향은 forward로 고정 // from/to tag를 사용하려면 노선마다, 지역마다 설정해야 함..ㅁ
query = f"""
[out:json];
area["name:en"="{area_search}"]->.searchArea;
relation["route"~"subway"](area.searchArea);
out meta;
>;
out body;
"""
subway_data = extract_osm_geodata(query)
route_elements = extract_route_elements(subway_data)
node_elements = extract_node_elements(subway_data)
G = create_route_graph_integrated(route_elements, node_elements)
m = create_folium_map_enhanced(G, zoom_start=12)
m.save(f"{area_search}_subway_map_{date}.html")
export_nodes_to_excel(G, filename=f"{area_search}_subway_nodes_{date}.xlsx")
export_edges_to_excel(G, filename=f"{area_search}_subway_edges_{date}.xlsx")
# print(G.nodes(data=True))  # 노드 정보 출력
# print(G.edges(data=True))  # 엣지 정보 출력
visualize_graph(G, f"Test_{area_search} Subway Network")
