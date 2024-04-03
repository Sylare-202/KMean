import random
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections as mc
import pylab as pl # matplotlib module
from math import atan2
import csv
import sys
import time
import argparse

# Default command to lunch program :
# python3 Kmeans.py -f -c 6 -i 36 -w  # To be able to run the program with the havy set of 2d data with 6 centroids and 36 iterations
# python3 Kmeans.py -l -w             # To be able to run the program with the light set of 2d data and default parameters
# python3 Kmeans.py -r -d <yourpath>  # To be able to run the program with the data from your .csv file (2d)
# python3 Kmeans.py -d3 -f 6 -i 36 -w # To be able to run the program with the havy set of 3d data with 6 centroids and 36 iterations

def main(DataFile, DataExport, n, k, WriteInFile, convHull, ThirdDim=False):
    start_time = time.time()
    
    if not ThirdDim:
        Data = ExtractData(DataFile, ThirdDim)
        print(f"--- {len(Data)} Data extracted from file ---\n")

        Centroids = oldCentroids = ConvexHull = []
        MinClusterVariance = sys.maxsize

        Iteration = 0
        while Iteration < n:
            while True:
                ClosestCentroids = tmpData = []
                if not Centroids:
                    Centroids = StartingCentroids(k, ThirdDim)

                ClosestCentroids = FindClosestCentroid(Data, Centroids, ThirdDim)
                for i in range(len(Data)):
                    tmpData.append([Data[i][0], Data[i][1], ClosestCentroids[i][0], ClosestCentroids[i][1]])

                Centroids = MeanCentroid(tmpData, Centroids, ThirdDim)

                if (Centroids == oldCentroids):
                    for i in range(len(Data)):
                        tmpData.append([Data[i][0], Data[i][1], ClosestCentroids[i][0], ClosestCentroids[i][1]])
                    break
                oldCentroids = Centroids

            ClusterVariance = SumClusterVariance(tmpData, Centroids, ThirdDim)
            if MinClusterVariance > ClusterVariance:
                print(f"\tNew Variance at iteration {Iteration}")
                BestVariance = tmpData
                MinClusterVariance = ClusterVariance
                Centro = Centroids
            
            Centroids = oldCentroids = []
            Iteration+=1

        if WriteInFile:
            WriteData(DataExport, BestVariance)

        if convHull:
            CentroidPoints = []
            for i in range(len(Centro)):
                tmpCentroids = []
                for j in range(len(BestVariance)):
                    if BestVariance[j][2] == Centro[i][0] and BestVariance[j][3] == Centro[i][1]:
                        tmpCentroids.append([BestVariance[j][0], BestVariance[j][1]])
                CentroidPoints.append(tmpCentroids)

            for i in range(len(Centro)):
                if CentroidPoints[i]:
                    ConvexHull.append(AlgoGraham(CentroidPoints[i]))
          
        print("\n--- %s seconds ---" % round((time.time() - start_time), 3))
        draw2D(BestVariance, ConvexHull)
    
    elif ThirdDim:
        Data = ExtractData(DataFile, ThirdDim)
        print(f"--- {len(Data)} Data extracted from file ---\n")
        
        Centroids = oldCentroids = []
        MinClusterVariance = sys.maxsize

        Iteration = 0
        while Iteration < n:
            while True:
                ClosestCentroids = tmpData = []
                if not Centroids:
                    Centroids = StartingCentroids(k, ThirdDim)

                ClosestCentroids = FindClosestCentroid(Data, Centroids, ThirdDim)
                for i in range(len(Data)):
                    tmpData.append([Data[i][0], Data[i][1], Data[i][2], ClosestCentroids[i][0], ClosestCentroids[i][1], ClosestCentroids[i][2]])

                Centroids = MeanCentroid(tmpData, Centroids, ThirdDim)

                if (Centroids == oldCentroids):
                    for i in range(len(Data)):
                        tmpData.append([Data[i][0], Data[i][1], Data[i][2], ClosestCentroids[i][0], ClosestCentroids[i][1], ClosestCentroids[i][2]])
                    break
                oldCentroids = Centroids

            ClusterVariance = SumClusterVariance(tmpData, Centroids, ThirdDim)
            if MinClusterVariance > ClusterVariance:
                print(f"\tNew Variance at iteration {Iteration}")
                BestVariance = tmpData
                MinClusterVariance = ClusterVariance

            Centroids = oldCentroids = []
            Iteration+=1

        if WriteInFile:
            WriteData(DataExport, BestVariance)
        print("\n--- %s seconds ---" % round((time.time() - start_time), 3))
        draw3D(BestVariance)


def CalcAngle(x1, x2 = None):
    if x2 == None:
        x2 = anchor # First anchor point
    x = x1[0] - x2[0]
    y = x1[1] - x2[1]
    return atan2(y, x) # Calculate the angle in radians

def DistanceHull(x1, x2 = None):
    if x2 == None:
        x2 = anchor # First anchor point
    x = x1[0] - x2[0]
    y = x1[1] - x2[1]
    return y**2 + x**2

def Deteminant(p1, p2, p3):
	return   (p2[0]-p1[0])*(p3[1]-p1[1]) \
			-(p2[1]-p1[1])*(p3[0]-p1[0])

def SortAlgo(a):
    if len(a) <= 1:
        return a
    Min, Equal, Max = [], [], []
    PivoAngle = CalcAngle(a[randint(0,len(a)-1)]) # select random pivot
    for point in a:
        
        PointAngle = CalcAngle(point) # Calcule the angle
        
        if PointAngle < PivoAngle:
            Min.append(point)
            
        elif PointAngle == PivoAngle:
            Equal.append(point)
            
        else:
            Max.append(point)
            
    return SortAlgo(Min) + sorted(Equal, key=DistanceHull) + SortAlgo(Max)

def SumClusterVariance(dataPoints, centeroids, thirdDim):
    returnSum = 0
    if not thirdDim:
        for center in centeroids:
            for point in dataPoints:
                if point[2] == center[0] and point[3] == center[1]:
                    if point[-2] == center[0] and point[-1] == center[1]:
                        returnSum += Distance2D(point[0], point[1], center[0], center[1])
    elif thirdDim:
        for center in centeroids:
            for point in dataPoints:
                if point[3] == center[0] and point[4] == center[1] and point[5] == center[2]:
                    if point[-3] == center[0] and point[-2] == center[1] and point[-1] == center[2]:
                        returnSum += Distance3D(point[0], point[1], point[2], center[0], center[1], center[2])
    return returnSum

# def SumClusterVariance_3d(dataPoints,, ThirdDim centeroids):
#     returnSum = 0
#     for center in centeroids:
#         for point in dataPoints:
#             if point[3] == center[0] and point[4] == center[1] and point[5] == center[2]:
#                 if point[-3] == center[0] and point[-2] == center[1] and point[-1] == center[2]:
#                     returnSum += Distance3D(point[0], point[1], point[2], center[0], center[1], center[2])
#     return returnSum

def FindClosestCentroid(Data, Centroids, tirdDim):
    tmpData = []
    if not tirdDim:
        for i in range(len(Data)):
            minDist = sys.maxsize
            closestCentroid = []
            for j in range(len(Centroids)):
                Distance = Distance2D(Data[i][0], Data[i][1], Centroids[j][0], Centroids[j][1])
                if Distance < minDist:
                    minDist = Distance
                    closestCentroid = Centroids[j]
            tmpData.append(closestCentroid)
    elif tirdDim:
        for i in range(len(Data)):
            minDist = sys.maxsize
            closestCentroid = []
            for j in range(len(Centroids)):
                Distance = Distance3D(Data[i][0], Data[i][1], Data[i][2], Centroids[j][0], Centroids[j][1], Centroids[j][2])
                if Distance < minDist:
                    minDist = Distance
                    closestCentroid = Centroids[j]
            tmpData.append(closestCentroid)
    return tmpData

def MeanCentroid(DataPoints, centroids, tirdDim):
    newCentroids = []
    if not tirdDim:
        for centroid in centroids:
            x = 0
            y = 0
            nb = 0
            for DataPoint in DataPoints:
                if DataPoint[2] == centroid[0] and DataPoint[3] == centroid[1]:
                    x += DataPoint[0]
                    y += DataPoint[1]
                    nb += 1
            if nb != 0:
                x /= nb
                y /= nb
                newCentroids.append([x, y])
    elif tirdDim:
        for centroid in centroids:
            x = 0
            y = 0
            z = 0
            nb = 0
            for DataPoint in DataPoints:
                if DataPoint[3] == centroid[0] and DataPoint[4] == centroid[1] and DataPoint[5] == centroid[2]:
                    x += DataPoint[0]
                    y += DataPoint[1]
                    z += DataPoint[2]
                    nb += 1
            if nb != 0:
                x /= nb
                y /= nb
                z /= nb
                newCentroids.append([x, y, z])        
    return newCentroids

def Distance2D(x1, y1, x2, y2):
    return (x2 - x1)**2 + (y2 - y1)**2

def Distance3D(x1, y1, z1, x2, y2, z2):
    return (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2

def StartingCentroids(k, thirdDim):
    Centroids = []
    if not thirdDim:
        for i in range(k):
            tmp = []
            tmp.append(random.randint(-80, 80))
            tmp.append(random.randint(-80, 80))
            Centroids.append(tmp)
    elif thirdDim:
        for i in range(k):
            tmp = []
            tmp.append(random.randint(-80, 80))
            tmp.append(random.randint(-80, 80))
            tmp.append(random.randint(-80, 80))
            Centroids.append(tmp)
    return Centroids

def AlgoGraham(points):
    global anchor # we take the smallest y value to get the anchor
    
    nbr = None
    
    for i,(x,y) in enumerate(points):
        if nbr == None or y < points[nbr][1]:
            nbr = i
        if y == points[nbr][1] and x < points[nbr][0]:
            nbr = i
            
    anchor=points[nbr] # Now we have the anchor set
    
    PointsSorte = SortAlgo(points) # Sort the points by polar angle
    del PointsSorte[PointsSorte.index(anchor)]
    
    hull = [anchor, PointsSorte[0]] # Set the anchor and point with smallest polar angle to the hull
    
    for s in PointsSorte[1:]: # For each point in the sorted list (exept the anchor)
        while Deteminant(hull[-2], hull[-1],s) <= 0: # if the determinant negative the clockwise turn and if 0 there are collinear points and if the determinant positive the counterclockwise turn
            del hull[-1] # Go back to the previous point
            if len(hull) < 2: break # If we have only one point in the hull we stop
        hull.append(s)
    return hull

def ExtractData(DataFile, thirdDim):
    with open(DataFile, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        tmpData = list(csvreader)
    tmpData.pop(0)
    Data = []
    if not thirdDim:
        for i in range(len(tmpData) - 1):
            Data.append([float(tmpData[i][0]), float(tmpData[i][1])])
    elif thirdDim:
        for i in range(len(tmpData) - 1):
            Data.append([float(tmpData[i][0]), float(tmpData[i][1]), float(tmpData[i][2])])
    return Data

# def ExtractData_3d(DataFile):
#     with open(DataFile, "r") as csvfile:
#         csvreader = csv.reader(csvfile)
#         tmpData = list(csvreader)
#     tmpData.pop(0)
#     Data = []
#     for i in range(len(tmpData) - 1):
#         Data.append([float(tmpData[i][0]), float(tmpData[i][1]), float(tmpData[i][2])])
#     return Data

def WriteData(DataFile, Data):
    with open(DataFile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "x_centroid", "y_centroid"])
        for i in range(len(Data)):
            writer.writerow([Data[i][0], Data[i][1], Data[i][2], Data[i][3]])

def ReadDataFromFile(DataFile):
    with open(DataFile, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        tmpData = list(csvreader)
    tmpData.pop(0)
    Data = []
    for i in range(len(tmpData) - 1):
        Data.append([float(tmpData[i][0]), float(tmpData[i][1]), float(tmpData[i][2]), float(tmpData[i][3])])
    draw2D(Data)

def draw2D(samples, hull=None, size=7, drawLinks=True):
    X, Y, links, centroids = [], [], [], set()
    for sample in samples:
        X.append(sample[0])
        Y.append(sample[1])
        if len(sample) == 4:
            links.append([sample[:2], sample[2:]])
            centroids.add((sample[2], sample[3]))
    centroids = sorted(centroids)
    random.seed(42)
    random.shuffle(centroids)
    centroids = { cent : centroids.index(cent) for cent in centroids }
    
    # Set the colors of the centroids
    colors = cm.rainbow(np.linspace(0, 1., len(centroids)))
    C = None
    if len(centroids) > 0:
        C = [colors[centroids[(sample[2], sample[3])]] for sample in samples]
    
    fig, ax = pl.subplots(figsize=(size, size))
    
    
    fig.suptitle('Visualisation de %d données' % len(samples), fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    if drawLinks:
        ax.add_collection(mc.LineCollection(links, colors=C, alpha=0.1, linewidths=1))
    ax.scatter(X, Y, c=C, alpha=0.5, s=10)
    for cent in centroids:
        ax.plot(cent[0], cent[1], c='black', marker='^', markersize=8)
    
    # Ajout du contour convexe
    if hull != None:
        for j in range(len(hull)):
            for i in range(1,len(hull[j])+1):
                if i == len(hull[j]):
                    i = 0
                X1 = hull[j][i-1]
                X2 = hull[j][i]
                ax.plot((X1[0], X2[0]), (X1[1], X2[1]), c="Black", linewidth=0.5)
   
    ax.autoscale()
    ax.margins(0.05)
    plt.savefig("data/kmeans.png")
    plt.show()

def draw3D(samples, size=10, drawLinks=False):
    X, Y, Z, links, centroids = [], [], [], [], set()
    for sample in samples:
        X.append(sample[0])
        Y.append(sample[1])
        Z.append(sample[2])
        if len(sample) == 6:
            links.append([sample[:3], sample[3:]])
            centroids.add((sample[3], sample[4], sample[5]))
    centroids = sorted(centroids)
    random.seed(42)
    random.shuffle(centroids)
    centroids = { cent : centroids.index(cent) for cent in centroids }
    
    colors = cm.rainbow(np.linspace(0, 1., len(centroids)))
    C = None
    if len(centroids) > 0:
        C = [ colors[centroids[(sample[3], sample[4], sample[5])]] for sample in samples ]
    
    fig, ax = pl.subplots(figsize=(size, size))
    ax = pl.axes(projection='3d')
    fig.suptitle('Visualisation de %d données' % len(samples), fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    if drawLinks:
        ax.add_collection(mc.LineCollection(links, colors=C, alpha=0.1, linewidths=1))
    ax.scatter3D(X, Y, Z, c=C, alpha=0.5, s=10)
    for cent in centroids:
        ax.plot3D(cent[0], cent[1], cent[2], c='black', marker='^', markersize=8)
    ax.autoscale()
    ax.margins(0.05)
    plt.savefig("data/kmeans.png")    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data file to use", default="data/mock_2d_data.csv")
    parser.add_argument("-o", "--output", help="Output file path (.csv)", default="data/2d_data_export.csv")
    parser.add_argument("-i", "--iterations", help="Number of iterations", default=4**2)
    parser.add_argument("-c", "--clusters", help="Number of clusters", default=4)
    parser.add_argument("-l", "--light", help="Use a light dataset", action="store_true")
    parser.add_argument("-f", "--full", help="Use the full dataset", action="store_true")
    parser.add_argument("-r", "--read", help="Read data from file", action="store_true")
    parser.add_argument("-w", "--write", help="Write data to file (default : true)", action="store_true")
    parser.add_argument("-d3", "--dimention_3d", help="Draw 3D", action="store_true")
    parser.add_argument("-ch", "--chull", help="Draw convex hull", action="store_true")
    
    args = parser.parse_args()
    
    if args.read:
        ReadDataFromFile(str(args.data))
    
    elif args.light:
        DataFile = "data/mock_2d_data.csv"
        main(DataFile, args.output, int(args.iterations), int(args.clusters), args.write, args.chull)
    
    elif args.full:
        DataFile = "data/2d_data.csv"
        main(DataFile, args.output, int(args.iterations), int(args.clusters), args.write, args.chull)
    
    elif args.dimention_3d:
        DataFile = "data/3d_data.csv"
        main(DataFile, args.output, int(args.iterations), int(args.clusters), args.write, args.chull, args.dimention_3d)
    
    else:
        main(args.data, args.output, int(args.iterations), int(args.clusters), args.write, args.chull, args.dimention_3d)