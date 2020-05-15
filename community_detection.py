from pyspark import SparkContext, SparkConf
import sys
import time
from itertools import combinations, permutations
import queue
import copy

filter_threshold = 6
input_file = 'ub_sample_data.csv'
betweenness_file = 'betweenness.txt'
community_file = 'community.txt'


# tools function
def insert_dict(dic, key, value):
    if key in dic.keys():
        dic[key].append(value)
    else:
        dic[key] = [value]
    return dic


# generate neighbors
def get_neighbors(edges):
    neighbors = dict()
    for v1, v2 in edges:
        neighbors = insert_dict(neighbors, v1, v2)
        neighbors = insert_dict(neighbors, v2, v1)
    return neighbors


# Step 1: run BFS
# Step 2: Label each node by the number of shortest paths that reach it from the root node
def BFS(neighbors, parent, label, line, level, Q):
    while not Q.empty():
        current_node = Q.get()
        line.append(current_node)
        neighs = neighbors[current_node]
        for neigh in neighs:
            if neigh not in parent[current_node]:
                if neigh in label.keys():
                    if level[neigh] != level[current_node]:
                        label[neigh] += label[current_node]
                        parent[neigh].append(current_node)
                else:
                    label[neigh] = label[current_node]
                    parent[neigh] = [current_node]
                    level[neigh] = level[current_node] + 1
                    Q.put(neigh)
    return parent, label, line


# Step 3: Calculate for each edge e, the sum over all nodes Y (of the fraction) of the
#         shortest paths from the root X to Y that go through edge e
def get_betwn_in_one_iter(parent, label, line):
    cross_num = dict()
    score_node = dict.fromkeys(label.keys(), 1)
    for v in line[::-1]:
        for pre in parent[v]:
            cross_num[tuple(sorted((v, pre), reverse=False))] = score_node[v] * (label[pre] / label[v])
            score_node[pre] += score_node[v] * (label[pre] / label[v])
    return cross_num


# Step 4: sum up all cross_num together and divided by 2 as betweenness
def sum_betwn(betweenness, cross_num):
    for key, value in cross_num.items():
        if key in betweenness.keys():
            betweenness[key] += value / 2
        else:
            betweenness[key] = value / 2
    return betweenness


# calculate betweenness
def get_betweenness(neighbors):
    betweenness = dict()
    for node in neighbors.keys():
        parent = dict()
        parent[node] = []
        label = dict()
        label[node] = 1
        level = dict()
        level[node] = 0
        line = []
        Q = queue.Queue()
        Q.put(node)
        parent, label, line = BFS(neighbors, parent, label, line, level, Q)
        cross_num = get_betwn_in_one_iter(parent, label, line)
        betweenness = sum_betwn(betweenness, cross_num)
    betweenness = sorted(betweenness.items(), key=lambda s: ((-1) * s[1], s[0]), reverse=False)
    return betweenness


# generate adjacent matrix
def get_matrix(vertices, neighbors):
    vertices.sort(reverse=False)
    A = [[0] * len(vertices) for i in range(len(vertices))]
    for v1, neighbor_list in neighbors.items():
        for v2 in neighbor_list:
            A[vertices.index(v1)][vertices.index(v2)] = 1
    return A


# get the edge number of the original graph
def get_m(neighbors):
    m = 0
    for v1, neighbor_list in neighbors.items():
        m += len(neighbor_list)
    return m / 2

# run BFS to generate community
def generate_community(neighbors, start_node):
    community = dict()
    visited = dict()
    visited[start_node] = 1
    Q = queue.Queue()
    Q.put(start_node)
    while not Q.empty():
        current_node = Q.get()
        if len(neighbors[current_node]) == 0:
            community[current_node] = []
            continue
        for neighbor in neighbors[current_node]:
            community = insert_dict(community, current_node, neighbor)
            if neighbor not in visited.keys():
                visited[neighbor] = 1
                Q.put(neighbor)
    return community

# initialize communities from original graph
def init_community(graph):
    community_rep = dict()
    community_key_list = []
    for v in graph.keys():
        sub_community = generate_community(graph, v)
        sub_comm_keys = tuple(sorted(sub_community.keys()))
        community_key_list.append(sub_comm_keys)
        community_rep[sub_comm_keys] = sub_community

    community_list = []
    for keys in set(community_key_list):
        community_list.append(community_rep[keys])
    return community_list

# detect communities
def detect_communities(old_community, v1, v2):
    community1 = generate_community(old_community, v1)
    # if there is still only one community in the whole graph
    if len(community1.keys()) == len(old_community.keys()):
        return [community1]
    else:
        community2 = generate_community(old_community, v2)
    return [community1, community2]

# remove the edge with the highest betweenness
def get_communities(old_community, edge_rm):
    (v1, v2) = edge_rm
    old_community[v1].remove(v2)
    old_community[v2].remove(v1)
    community_list = detect_communities(old_community, v1, v2)
    return community_list

# calculate the Modularity Q score
def get_Q(community_list, A, m, vertices, neighbors):
    Q = 0
    for community in community_list:
        potential_edges = permutations(community.keys(), 2)
        for (v1, v2) in potential_edges:
            ki = len(neighbors[v1])
            kj = len(neighbors[v2])
            index_v1 = vertices.index(v1)
            index_v2 = vertices.index(v2)
            Q += (A[index_v1][index_v2] - ki * kj / (2 * m))
    return Q / (2 * m)


if __name__ == "__main__":
    start_time = time.time()

    conf = SparkConf().setAppName('inf553_hw4_task2').setMaster('local[3]')
    sc = SparkContext(conf=conf)

    ub = sc.textFile(input_file) \
        .filter(lambda s: s != 'user_id,business_id') \
        .distinct() \
        .map(lambda s: str(s).split(',')) \
        .map(lambda s: (s[1], s[0])) \
        .groupByKey() \
        .mapValues(list) \
        .flatMap(lambda s: list(combinations(sorted(s[1], reverse=False), 2))) \
        .map(lambda s: (s, 1)) \
        .reduceByKey(lambda u, v: u + v) \
        .filter(lambda s: s[1] >= filter_threshold) \
        .map(lambda s: s[0]) \
        .persist()

    vertices = ub.flatMap(lambda s: s) \
        .distinct() \
        .map(lambda s: s) \
        .collect()

    edges = ub.collect()
    # generate neighbors
    neighbors = get_neighbors(edges)
    betweenness = get_betweenness(neighbors)

    with open(betweenness_file, 'w') as f:
        for key, value in betweenness:
            f.write(str(key) + ', %f\n' % value)

    # generate adjacent matrix
    A = get_matrix(vertices, neighbors)
    m = get_m(neighbors)

    # # remove the edge with the highest betweenness
    # community_list = get_communities(neighbors, betweenness[0][0])

    # initialize communities
    community_list = init_community(neighbors)

    # calculate the Modularity Q score
    Q = get_Q(community_list, A, m, vertices, neighbors)
    result = [Q, community_list]

    community_evaluation = []
    for comm in community_list:
        community_evaluation.append([comm, get_betweenness(comm)[0]])

    # len(community_list) < len(vertices)
    while len(community_list) < len(vertices):
        community_evaluation.sort(key=lambda x: x[1][1], reverse=True)

        comm_divided = community_evaluation[0]
        new_list = get_communities(comm_divided[0], comm_divided[1][0])
        for comm in new_list:
            if len(comm.keys()) == 1:
                community_evaluation.append([comm, [None, 0]])
            else:
                community_evaluation.append([comm, get_betweenness(comm)[0]])
            community_list.append(comm)

        community_evaluation.remove(comm_divided)
        community_list.remove(comm_divided[0])

        Q = get_Q(community_list, A, m, vertices, neighbors)

        if result[0] < Q:
            result[0] = Q
            result[1] = copy.deepcopy(community_list)

    result = [sorted(community.keys(), reverse=False) for community in result[1]]

    with open(community_file, 'w') as f:
        result.sort(key=lambda s: (len(s), s), reverse=False)
        for community in result:
            f.write(str(community)[1:-1] + '\n')

    print('Duration: %s' % (time.time() - start_time))
