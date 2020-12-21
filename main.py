import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import KernelPCA
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from graph import Graph
from util import make_ground_truth


def read_egonets(path='./learning-social-circles/egonets',egonet='239.egonet'):
    i=0
    temp={}
    # path = './egonets'
    # files = sorted(os.listdir(path), key = lambda x:int(x[:-7]))
    # for egonet in files:
    egonet = str(egonet)+'.egonet'
    ego = int(egonet[:-7])
    with open(os.path.join(path,egonet), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[-1]=='\n':
                line=line[:-1]
            if line=='\n' or line=='':
                break
            x = line.split(' ')[0][:-1]
            # print(x)
            if int(x) <= ego:
                continue
            y = line.split(' ')[1:]
            if y==['']:
                temp[str(x)]=set()
            else:
                # y = [int(ele) for ele in y]
                y = [int(ele) for ele in y if int(ele)>ego]
                temp[str(x)]=set(y)
    adjacency_matrix = np.zeros((len(temp), len(temp)))
    # print(adjacency_matrix.shape)
    for i in range(len(temp)):
        adjacency_matrix[i][i] = 1
        for ele in temp[str(i+ego+1)]:
            adjacency_matrix[i][ele-ego-1] = 1
    """
    Step 1: Compute an approximation of the exponential adjacency matrix E of the friend graph
    """
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
    """
    Step 2: Apply a generic clustering algorithm
    """
    # transformer = KernelPCA(n_components=50, kernel='rbf', n_jobs=-1)
    # kpca = transformer.fit_transform(matrix)

    # # Clustering
    pred = SpectralClustering(n_clusters=n, random_state=0).fit(matrix)
    return pred.labels_

def clean_up_circle(ego, pred, n, edges, low_constrain):
    """
    Step 3: Throw away low density circles
    """
    circles = [set() for i in range(n)]
    start = ego + 1
    for i, ele in enumerate(pred):
        circles[int(ele)].add(i + start)
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
        elif not len(circles[i]):
            del circles[i]
            n-=1
        else:
            i+=1
    return circles
    
def augment_circle(circles, F, edges):
    """
    Step 4: Augment the remaining circles by adding in people with more than F friends in the circle
    """
    counts = np.zeros((len(edges), len(circles)))
    for i, ele in enumerate(edges):
        for edge in edges[str(ele)]:
            for j, circle in enumerate(circles):
                if edge in circle:
                    counts[i][j] += 1
    for i, ele in enumerate(edges):
        for j in range(counts.shape[1]):
            count = counts[i][j]
            if count > F or count > len(circles[j]) * 0.5:
                circles[j].add(int(ele))
    return circles
    
def connected_components(ego, circles, edges, members_lower_bound=5, members_upper_bound=15):
    """
    Step 5: Include small connected components as independent circles
    """
    new_circles = []
    for circle in circles:
        # print("Circle:", circle)
        g = Graph(len(edges))
        for ele in circle:
            for edge in edges[str(ele)]:
                if edge in circle:
                    g.addEdge(ele - ego - 1, edge - ego - 1)
        cc = g.connectedComponents()
        i = 0
        n = len(cc)
        while(i < n):
            count = 0
            if len(cc[i]) == 1:
                del cc[i]
                n-=1
            else:
                i+=1
        for i in range(len(cc)):
            for j in range(len(cc[i])):
                cc[i][j] += (ego + 1)
        # print(cc)
        for component in cc:
            if len(component) >= members_lower_bound and len(component) <= members_upper_bound:
                # print(component)
                new_circles.append(set(component))
                for x in component:
                    circle.remove(x)
        if len(circle):
            new_circles.append(circle)
    return new_circles
    
def merge_circles(circles, portion=0.75):
    """
    Step 6: Merge circles with more than 75% overlap
    """
    new_circles = []
    for i in range(len(circles)):
        hasUnion = False
        for j in range(i+1, len(circles)):
            if i == j:
                continue
            union = circles[i].union(circles[j])
            intersection = circles[i].intersection(circles[j])
            if len(intersection) / len(union) > portion:
                new_circles.append(union)
                hasUnion = True
                break
        if not hasUnion:
            new_circles.append(circles[i])
    return new_circles

def save_result(ego,circles):
    result = open(os.path.join('./result',str(ego)+'.result'),'w')
    result.write("UserId,Predicted\n")
    # print("UserId,Predicted")
    # print(str(ego)+",",end="")
    result.write(str(ego)+",")
    circles = [list(i) for i in circles]
    #print(*circles, sep=';')
    for circle in circles:
        # print(*circle,end='')
        for idx in circle:
            result.write(str(idx))
            if idx != circle[-1]:
                result.write(" ")
        if circle!= circles[-1]:
            # print(";",end='')
            result.write(";")
    # print('\n')
    result.write('\n')
    result.close()

def generate_circles(path='./learning-social-circles/egonets',egonet='239.egonet'):
    ego, exp_adjacency_matrix, edges = read_egonets(egonet=egonet)
    n_circles = 6
    pred = clustering(exp_adjacency_matrix, n_circles)
    low_constrain = 0.2
    circles = clean_up_circle(ego, pred, n_circles, edges, low_constrain)
    # print(circles)
    F = 8
    circles = augment_circle(circles, F, edges)
    circles = connected_components(ego, circles, edges)
    circles = merge_circles(circles)
    # print(circles)
    save_result(ego,circles)

if __name__ == '__main__':
    files = sorted(os.listdir("./learning-social-circles/egonets/"), key = lambda x:int(x[:-7]))
    # files = sorted(os.listdir("./learning-social-circles/Training/"), key = lambda x:int(x[:-8]))
    files = [int(file[:-7]) for file in files]
    # print(*files,sep=" ")
    for ego in files:
        print(ego)
        try:
            generate_circles(egonet=ego)
        except:
            print("Egonet of " + str(ego) + " is not valid!!")
        try:
            make_ground_truth(ego)
        except:
            print("No file: " + str(ego) +'.circles!!')