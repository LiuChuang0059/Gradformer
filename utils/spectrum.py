import torch
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs


def spectral_distance_matrix(edge_index, k=2):
    """
    计算图中所有节点之间的谱距离矩阵。
    edge_index: 边的索引
    k: 用于谱嵌入的特征向量数量
    """
    # 创建 NetworkX 图对象
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # 获取图中的所有节点
    nodes = list(G.nodes)

    # 初始化一个空的tensor来保存谱距离矩阵
    num_nodes = len(nodes)
    distance_matrix = torch.zeros((num_nodes, num_nodes))

    # 计算所有节点之间的谱距离
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance_matrix[i, j] = spectral_distance(G, nodes[i], nodes[j], k)
            distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix


def spectral_distance(G, node1, node2, k=2):
    """
    计算图G中两个节点的谱距离。
    G: NetworkX图对象
    node1, node2: 要计算距离的节点
    k: 用于谱嵌入的特征向量数量
    """
    # 计算拉普拉斯矩阵
    L = nx.laplacian_matrix(G).asfptype()

    # 计算前k个最小的非零特征值和对应的特征向量
    vals, vecs = eigs(L, k=k + 1, which='SM')
    vecs = vecs[:, 1:k + 1]  # 排除第一个特征值（为0）

    # 谱嵌入：每个节点映射到k维空间
    spectral_embedding = {node: vecs[i, :] for i, node in enumerate(G.nodes())}

    # 计算并返回两个节点的欧几里得距离
    return np.linalg.norm(spectral_embedding[node1] - spectral_embedding[node2])