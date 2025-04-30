# -*- coding: utf-8 -*-
#hit-2022113574-lab1 of master
#日期：2025/4/30                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
import re
import random
import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# ====================== 原有函数（完全保持不变） ======================

def read_and_clean_text(file_path):
    """读取文件并进行清洗"""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

def build_graph(text):
    """构建有向图"""
    words = text.split()
    graph = defaultdict(lambda: defaultdict(int))
    
    if len(words) < 2:
        print("Warning: Text is too short to build a meaningful graph!")
        return graph
    
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        graph[word1][word2] += 1
    return graph

def visualize_graph(graph, highlight_paths=None):
    """可视化图，可突出显示特定路径，并显示边的权值"""
    if not graph:
        print("Graph is empty, nothing to visualize!")
        return
        
    G = nx.DiGraph()
    nodes = list(graph.keys())
    
    for node in nodes:
        for neighbor, weight in graph[node].items():
            G.add_edge(node, neighbor, weight=weight)
    
    if not G.edges():
        print("No edges to visualize!")
        return
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue', edgecolors='navy')
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # 显示边的权值
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    if highlight_paths:
        if not isinstance(highlight_paths[0], list):
            highlight_paths = [highlight_paths]
        
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, path in enumerate(highlight_paths):
            if len(path) < 2:
                continue
            edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, 
                                 edge_color=colors[i % len(colors)], 
                                 width=3.0, 
                                 arrowsize=20)
            nx.draw_networkx_nodes(G, pos, nodelist=path, 
                                 node_size=1000, 
                                 node_color=colors[i % len(colors)])
    
    plt.title("Word Graph Visualization" + 
             (" (Highlighted Paths)" if highlight_paths else ""))
    plt.show()

def find_bridge_words(graph, word1, word2):
    """查找word1和word2之间的桥接词"""
    # 检查输入单词是否在图中
    missing_words = []
    if word1 not in graph:
        missing_words.append(f"'{word1}'")
    if word2 not in graph:
        missing_words.append(f"'{word2}'")
    
    if missing_words:
        print(f"No {missing_words[0]} in the graph!" if len(missing_words) == 1 
              else f"No {missing_words[0]} and {missing_words[1]} in the graph!")
        return

    bridge_words = []
    # 查找所有满足条件的桥接词word3
    for word3 in graph[word1]:
        if word2 in graph[word3]:
            bridge_words.append(word3)

    if bridge_words:
        if len(bridge_words) == 1:
            print(f"The bridge word from '{word1}' to '{word2}' is: {bridge_words[0]}.")
        else:
            print(f"The bridge words from '{word1}' to '{word2}' are: {', '.join(bridge_words)}.")
    else:
        print(f"No bridge words from '{word1}' to '{word2}'!")

def find_bridge_words_new(graph, word1, word2):
    """查找word1和word2之间的桥接词"""
    if word1 not in graph or word2 not in graph:
        return None
    return [word3 for word3 in graph[word1] if word2 in graph[word3]]
def process_new_text_with_bridge_words(graph, new_text):
    """处理新文本，在相邻单词间插入bridge words"""
    words = new_text.lower().split()
    if len(words) < 2:
        return new_text  # 单字文本无需处理
    
    result = [words[0]]
    for i in range(len(words)-1):
        word1 = words[i]
        word2 = words[i+1]
        
        bridge_words = find_bridge_words_new(graph, word1, word2)
        if bridge_words:
            # 随机选择一个bridge word插入
            chosen_bridge = random.choice(bridge_words)
            result.append(chosen_bridge)
        result.append(word2)
    
    # 保留原始文本的大小写和标点（简单实现）
    return ' '.join(result).capitalize()


def find_shortest_path(graph, word1, word2=None):
    """计算最短路径"""
    if word1 not in graph:
        print(f"Word '{word1}' not found in graph!")
        return None
    
    G = nx.DiGraph()
    for node in graph:
        for neighbor, weight in graph[node].items():
            G.add_edge(node, neighbor, weight=weight)
    
    if word2:
        if word2 not in graph:
            print(f"Word '{word2}' not found in graph!")
            return None
        
        try:
            all_paths = list(nx.all_shortest_paths(G, word1, word2, weight='weight'))
            path_length = nx.shortest_path_length(G, word1, word2, weight='weight')
            
            print(f"\nFound {len(all_paths)} shortest path(s) from '{word1}' to '{word2}'")
            print(f"Path length: {path_length}")
            
            for i, path in enumerate(all_paths[:3]):
                print(f"Path {i+1}: {' → '.join(path)}")
            
            #visualize_graph(graph, highlight_paths=all_paths[0])
            return all_paths
        except nx.NetworkXNoPath:
            print(f"\nNo path exists from '{word1}' to '{word2}'!")
            return None
    else:
        print(f"\nCalculating shortest paths from '{word1}' to all other words...")
        paths = nx.single_source_dijkstra_path(G, word1, weight='weight')
        lengths = nx.single_source_dijkstra_path_length(G, word1, weight='weight')
        
        for target, path in paths.items():
            if target == word1:
                continue
            print(f"\nTo '{target}':")
            print(f"Path: {' → '.join(path)}")
            print(f"Length: {lengths[target]}")
        
        return paths

def input_and_find_bridge_words(graph):
    """封装输入和查找桥接词的完整流程"""
    while True:
        print("\n" + "="*50)
        print("Bridge Word Finder (Enter 'q' to quit)")
        word1 = input("请输入第一个单词 (word1): ").lower().strip()
        if word1 == 'q':
            break
            
        word2 = input("请输入第二个单词 (word2): ").lower().strip()
        if word2 == 'q':
            break
            
        if not word1 or not word2:
            print("Error: Input cannot be empty!")
            continue
            
        bridge_words = find_bridge_words(graph, word1, word2)
        
        if bridge_words is not None:
            if bridge_words:
                if len(bridge_words) == 1:
                    print(f"\nThe bridge word from '{word1}' to '{word2}' is: {bridge_words[0]}")
                else:
                    print(f"\nThe bridge words from '{word1}' to '{word2}' are: {', '.join(bridge_words)}")
            else:
                print(f"\nNo bridge words from '{word1}' to '{word2}'!")


def process_user_text(graph):
    """处理用户输入的新文本"""
    while True:
        print("\n" + "="*50)
        print("New Text Processor (Enter 'q' to quit)")
        user_input = input("请输入要处理的新文本: ").strip()
        
        if user_input.lower() == 'q':
            break
            
        if not user_input:
            print("Error: Input cannot be empty!")
            continue
            
        processed_text = process_new_text_with_bridge_words(graph, user_input)
        print("\n处理后的文本:")
        print(processed_text)

def find_shortest_path_interactive(graph):
    """最短路径交互界面"""
    while True:
        print("\n" + "="*50)
        print("Shortest Path Finder (Enter 'q' to quit)")
        word1 = input("请输入起始单词 (word1): ").lower().strip()
        if word1 == 'q':
            break
            
        if not word1:
            print("Error: Input cannot be empty!")
            continue
            
        word2 = input("请输入目标单词 (留空则计算到所有单词的路径): ").lower().strip()
        if word2 == 'q':
            break
            
        find_shortest_path(graph, word1, word2 if word2 else None)

# ====================== 新增PageRank相关函数 ======================

def compute_pagerank(graph, d=0.85, max_iter=100, tol=1e-6, initial_weights=None):
    """
    修正后的PageRank计算函数
    使用 {word: {neighbor: count}} 格式的输入图，其中 count 表示边的权重。
    计算包括对悬挂节点的处理。
    :param graph: 输入图 {source_node: {target_node: weight}}，weight 是边的权重（例如 count）
    :param d: 阻尼系数(通常0.85)
    :param max_iter: 最大迭代次数
    :param tol: 收敛阈值
    :param initial_weights: 初始权重(字典 {node: weight_value}，可选)
    :return: 排序后的(node, pagerank)列表
    """
    if not graph:
        return []

    # 获取所有节点：包括作为源节点的节点和作为目标节点的节点
    nodes = set(graph.keys())
    for targets in graph.values():
        nodes.update(targets.keys())
    nodes = list(nodes) # 转换为列表以便排序和索引（虽然这里主要按名称访问）
    N = len(nodes)

    if N == 0: # 如果图中没有任何节点
        return []

    # 初始化PR值
    pr = {}
    if initial_weights:
        # 归一化初始权重，并应用于所有节点（如果某个节点不在initial_weights中，默认为0）
        total_initial_weight = sum(initial_weights.values()) if initial_weights else 0
        if total_initial_weight > 0:
             pr = {n: initial_weights.get(n, 0) / total_initial_weight for n in nodes}
        else: # 如果初始权重为空或总和为0，则均匀初始化
             pr = {n: 1.0 / N for n in nodes}
    else:
        # 均匀初始化所有节点的PR值
        pr = {n: 1.0 / N for n in nodes}

    # 计算每个节点的出度 (指向其他节点的总权重)
    # 只对作为源节点的节点计算出度，目标节点如果不是源节点，其出度自然为0
    out_degree = {n: sum(graph.get(n, {}).values()) for n in nodes} # 使用 .get(n, {}) 处理只作为目标的节点


    # 找出悬挂节点 (出度为0的节点)
    # 悬挂节点是 graph 的 key 但 sum(graph[n].values()) == 0 的节点
    # 或者根本不是 graph 的 key (只作为目标节点)
    # 实际上，out_degree 的计算已经覆盖了所有 nodes。悬挂节点就是 out_degree[n] == 0 的节点。
    dangling_nodes = [n for n in nodes if out_degree[n] == 0]

    # 悬挂节点PR值的总和将均匀分发给所有节点
    # dangling_mass = sum(old_pr[n] for n in dangling_nodes)
    # 每个节点从悬挂节点获得的贡献是 d * dangling_mass / N

    for iteration in range(max_iter):
        old_pr = pr.copy()
        new_pr = {}
        
        # 计算悬挂节点在当前迭代中的PR总和
        dangling_mass = sum(old_pr[n] for n in dangling_nodes)

        # 计算新的PR值
        for node in nodes: # 计算每个节点的 PageRank
            # 来自正常链接的贡献 (来自非悬挂节点)
            incoming_pr_sum = 0
            # 遍历所有可能的源节点 neighbor
            for neighbor in nodes: # neighbor 是可能的源节点
                 # 检查 neighbor 是否是 graph 的 key 且有指向 node 的边
                if neighbor in graph and node in graph[neighbor]:
                    # 确保源节点 neighbor 不是悬挂节点 (出度 > 0)
                    if out_degree[neighbor] > 0:
                        # PageRank(neighbor) / out_degree(neighbor) * weight(neighbor -> node)
                        # = PageRank(neighbor) * (weight(neighbor -> node) / out_degree(neighbor))
                        incoming_pr_sum += old_pr[neighbor] * graph[neighbor][node] / out_degree[neighbor]

            # 计算新PR值
            # PR(node) = (1-d)/N + d * (来自正常链接的贡献) + d * (来自悬挂节点的贡献)
            # 来自悬挂节点的贡献 = dangling_mass / N
            new_pr[node] = (1 - d) / N + d * (incoming_pr_sum + dangling_mass / N)

        # 检查收敛性
        diff = sum(abs(new_pr[n] - old_pr[n]) for n in nodes)
        # print(f"Iteration {iteration+1}, Diff: {diff}") # Debugging line
        if diff < tol:
            break

        pr = new_pr # 更新PR值进行下一轮迭代

    # 迭代结束后，PR值的总和应该接近 1。最终归一化以确保总和为 1。
    total_pr_sum = sum(pr.values())
    if total_pr_sum > 0:
         normalized_pr = {k: v / total_pr_sum for k, v in pr.items()}
    else: # 避免除以零，虽然在非空图上不应该发生
         normalized_pr = pr


    # 按 PR 值排序并返回 (word, pagerank) 列表
    # 使用 items() 获取键值对，key=lambda x: -x[1] 按值降序排序
    return sorted(normalized_pr.items(), key=lambda item: -item[1])



def compute_tfidf_weights(text):
    """计算TF-IDF权重作为初始PR值"""
    sentences = [text]
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    return {word: score+0.1 for word, score in zip(feature_names, tfidf_scores)}

def visualize_pagerank(pagerank, top_n=10):
    """修正后的可视化函数"""
    top = pagerank[:top_n]
    words = [w for w, _ in top]
    values = [v for _, v in top]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(words[::-1], values[::-1], color='skyblue')
    plt.xlabel('PageRank Value')
    plt.title(f'Top {top_n} Words by PageRank')
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                va='center', ha='left')
    
    plt.tight_layout()
    plt.show()

def pagerank_analysis(graph, text=None):
    """PageRank分析交互界面"""
    print("\nPageRank Analysis Options:")
    print("1. Standard PageRank")
    print("2. TF-IDF Weighted PageRank")
    choice = input("Select method (1-2): ").strip()
    
    if choice == '1':
        print("\nComputing standard PageRank...")
        pagerank = compute_pagerank(graph)
    elif choice == '2':
        if not text:
            print("TF-IDF requires original text!")
            return
        print("\nComputing TF-IDF weighted PageRank...")
        tfidf_weights = compute_tfidf_weights(text)
        pagerank = compute_pagerank(graph, initial_weights=tfidf_weights)
    else:
        print("Invalid choice!")
        return
    
    if not pagerank:
        print("No PageRank results!")
        return
    
    print("\nTop 20 Words by PageRank:")
    print("="*40)
    print(f"{'Word':<15}{'PageRank':>15}")
    print("-"*40)
    for word, score in pagerank[:20]:
        print(f"{word:<15}{score:>15.4f}")
    
    visualize_pagerank(pagerank)

def random_walk(graph):
    """
    随机游走功能
    1. 随机选择起点
    2. 沿出边随机遍历，直到出现重复边或没有出边
    3. 用户可随时停止
    4. 将路径保存到文件
    """
    if not graph:
        print("Graph is empty, cannot perform random walk!")
        return
    
    # 随机选择起点
    current_node = random.choice(list(graph.keys()))
    print(f"\nStarting random walk from node: '{current_node}'")
    
    path = [current_node]
    visited_edges = set()
    stop_walk = False
    
    while True:
        # 检查是否有出边
        neighbors = list(graph[current_node].keys())
        if not neighbors:
            print(f"\nNode '{current_node}' has no outgoing edges, stopping walk.")
            break
        
        # 随机选择下一个节点
        next_node = random.choice(neighbors)
        edge = (current_node, next_node)
        
        # 检查是否重复边
        if edge in visited_edges:
            print(f"\nRepeated edge detected: '{current_node}' -> '{next_node}', stopping walk.")
            break
        visited_edges.add(edge)
        
        path.append(next_node)
        current_node = next_node
        
        # 显示当前路径
        print(f"\nCurrent path: {' → '.join(path)}")
        
        # 用户可选择继续或停止
        user_input = input("Press Enter to continue, or 'stop' to end walk: ").strip().lower()
        if user_input == 'stop':
            print("\nRandom walk stopped by user.")
            break
    
    # 保存路径到文件
    path_str = ' '.join(path)
    with open("random_walk.txt", "w", encoding="utf-8") as f:
        f.write(path_str)
    print(f"\nRandom walk path saved to 'random_walk.txt'")
    print(f"Final path: {path_str}")


def main():
    #file_path = "./Easy Test.txt"
    file_path = input("请输入文本文件路径：")  # 获取文件路径
    try:
        print("Loading and processing text file...")
        text = read_and_clean_text(file_path)
        
        if not text:
            print("Error: Text is empty after cleaning!")
            return
            
        print(f"\nCleaned text sample (first 100 chars):\n{text[:100]}...")
        
        graph = build_graph(text)
        if not graph:
            print("Error: Failed to build graph!")
            return
            
        print(f"\nGraph contains {len(graph)} nodes.")
        print("Sample nodes:", list(graph.keys())[:10])
        
        # 更新主菜单（新增随机游走选项）
        while True:
            print("\n" + "="*50)
            print("Main Menu")
            print("1. Find bridge words between two words")
            print("2. Process new text with bridge words")
            print("3. Find shortest path between words")
            print("4. Compute PageRank")
            print("5. Visualize word graph")
            print("6. Random Walk (New)")
            print("7. Exit")
            choice = input("请选择功能 (1-7): ").strip()
            
            if choice == '1':
                input_and_find_bridge_words(graph)
            elif choice == '2':
                process_user_text(graph)
            elif choice == '3':
                find_shortest_path_interactive(graph)
            elif choice == '4':
                pagerank_analysis(graph, text)
            elif choice == '5':
                visualize_graph(graph)
            elif choice == '6':
                random_walk(graph)
            elif choice == '7':
                print("Goodbye!")
                break
            else:
                print("Invalid choice, please try again!")
                
    except FileNotFoundError:
        print(f"\nError: File '{file_path}' not found!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()