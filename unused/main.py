import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
# friends = {}
from sklearn.decomposition import KernelPCA
egonets = np.array([])
temp={}

def read_egonets(path=None):
    i=0
    path = './learning-social-circles/egonets'
    files = ['23978.egonet']
    # path = './egonets'
    # files = sorted(os.listdir(path), key = lambda x:int(x[:-7]))
    # for egonet in files:
    egonet = files[0]
    ego = int(egonet[:-7])
    # print(ego)
    with open(os.path.join(path,egonet), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[-1]=='\n':
                line=line[:-1]
            if line=='\n' or line=='':
                break
            x = line.split(' ')[0][:-1]
            # print(x)
            y = line.split(' ')[1:]
            if y==['']:
                temp[str(x)]=set()
            else:
                y = [int(ele) for ele in y]
                temp[str(x)]=set(y)
    adjacency_matrix = np.zeros((len(temp), len(temp)))
    # print(adjacency_matrix.shape)
    # print(ego)
    for i in range(len(temp)):
        adjacency_matrix[i][i] = 1
        if str(i+ego+1) not in temp:
            continue
        for ele in temp[str(i+ego+1)]:
            adjacency_matrix[i][ele-ego-1] = 1
    
    exp_adjacency_matrix = adjacency_matrix
    degree_adjacency_matrix = adjacency_matrix
    n = 5
    factorial = 1
    for i in range(2, min(len(temp), n)+1):
        factorial*=i
        degree_adjacency_matrix = np.dot(degree_adjacency_matrix,adjacency_matrix)/factorial
        exp_adjacency_matrix += degree_adjacency_matrix
    return ego, exp_adjacency_matrix, temp

def clustering(matrix, n):
    # transformer = KernelPCA(n_components=50, kernel='rbf', n_jobs=-1)
    # kpca = transformer.fit_transform(matrix)

    # # Clustering
    print('')
    pred = SpectralClustering(n_clusters=n, random_state=0).fit(matrix)
    return pred.labels_

def build_circle(ego, predn, n):
    circles = [set() for i in range(n)]
    start = ego + 1
    for i, ele in enumerate(pred):
        # print(i,ele)
        circles[int(ele)].add(i + start)
    return circles

def clean_up_circle(circles, n, edges, low_constrain):
    i = 0
    while(i < n):
        count = 0
        l = len(circles[i])
        for ele in circles[i]:
            for edge in edges[str(ele)]:
                if edge in circles[i]:
                    count+=1
        assert(count%2==0)
        if (count/2 < low_constrain * (l * (l - 1) / 2)):
            del circles[i]
            n-=1
        else:
            i+=1
    return circles

def augment_circle(ego,trim_circles,F,friend_list):
    for circle in trim_circles:
        for index, friends in friend_list.items():
            # print(int(index),friends)
            count = 0
            for friend in friends:
                if friend in circle:
                    count += 1
            if count >= F or count >= len(circle)/2:
                circle.add(int(index))
    return trim_circles

def check_overlap(A,B,percentage):
    overlap = A.intersection(B)
    if max(len(overlap)/len(A),len(overlap)/len(B)) >= percentage:
        return True
    return False

def merge(circles,percentage):
    i = 0
    j = 1
    while(i < len(circles)-1):
        if check_overlap(circles[i],circles[j],percentage):
            circles[i] = circles[i].union(circles[j])
            del circles[j]
            j = i+1
        else:
            j += 1
        if j==len(circles)-1:
            i+=1
            j = i+1
    
    return circles


ego, exp_adjacency_matrix, edges = read_egonets()
n_circles = 6
pred = clustering(exp_adjacency_matrix, n_circles)
# print(pred)
circles = build_circle(ego,pred,n_circles)
# print(circles)
low_constrain = 0.2
trim_circles = clean_up_circle(circles, n_circles, edges, low_constrain)
print(trim_circles)
#print(temp)
augmented_circles = augment_circle(ego,trim_circles,6,temp)
print(augmented_circles)
P = 0.75
result = merge(augmented_circles,P)