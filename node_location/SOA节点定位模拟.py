"""
File  : SOA节点定位模拟.py
Author: ACHIEVE_DREAM
Date  : 2023/12/4
"""
import numpy as np

from SOA import SOA
import matplotlib.pyplot as plt


def distance2(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def fitness_func(population: np.ndarray, **kwargs):
    """
    适应度函数
    :param population: 种群
    :return: 适应度
    """
    # 锚节点坐标列表, 已知
    anchor_nodes: np.ndarray = kwargs['anchor_nodes']
    # 从anchor_nodes中随机选择1/3的锚节点作为与待测节点连接的锚节点, replace=False代表不可重复选择
    connected_anchor_nodes = anchor_nodes[
        np.random.choice(anchor_nodes.shape[0], anchor_nodes.shape[0] // 3, replace=False)]
    # connected_anchor_nodes = anchor_nodes[:2]  # 为了方便测试, 只取前两个锚节点
    # 待测节点的真实位置, 主要用于计算距离, 实际应用中不需要此参数, 而是需要rssi计算得到的与锚节点的距离即可
    unknown_real_node = kwargs['unknown_node']
    # 计算每个个体的适应度
    fitness = np.empty(population.shape[0])
    for i, individual in enumerate(population):
        for j, anchor in enumerate(connected_anchor_nodes):
            # 此处是模拟, 真实情况的距离应该用rssi来计算
            fitness[i] += (abs(distance2(unknown_real_node, anchor) - distance2(individual, anchor))) ** 0.5
    return fitness


if __name__ == '__main__':
    # 模拟的节点位置, 边界为(0, 100), 一共100个节点
    nodes = np.random.uniform(0, 100, (100, 2))
    # 锚节点取20个
    anchor_nodes = nodes[:20]
    # 位置节点取80个
    unknown_nodes = nodes[20:]
    # 待测的未知节点
    unknown_node = unknown_nodes[6]
    soa = SOA(2, 100, 20, 0, 100, fitness_func, anchor_nodes=anchor_nodes, unknown_node=unknown_node)
    best_solution, best_fitness, gene_best_fitness = soa.snake_optimization()
    print(f"估计位置:{best_solution}\n真实位置:{unknown_node}\n适应度:{best_fitness}")
    plt.plot(gene_best_fitness)
    plt.title("SOA算法")
    plt.xlabel("迭代次数")
    plt.ylabel("适应度")
    plt.show()
