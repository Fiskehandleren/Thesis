from haversine import haversine, Unit
import numpy as np
import time
from dataloader import load_data
from tqdm import tqdm
# compute the haversine distance between each station and all other stations
def compute_distance_matrix(df):
    t1 = time.time()
    clusters = df['Cluster'].unique()
    clusters = clusters[clusters != "SHERMAN"]
    distance_matrix = np.zeros((len(clusters), len(clusters)))
    for i in tqdm(range(len(clusters))):
        for j in range(len(clusters)):
            if i == j:
                continue
            elif distance_matrix[j][i] != 0:
                distance_matrix[i][j] = distance_matrix[j][i]
            else:

                distance_matrix[i][j] = haversine(
                    (df[df['Cluster'] == clusters[i]]['Latitude'].mean(), df[df['Cluster'] == clusters[i]]['Longitude'].mean()), 
                    (df[df['Cluster'] == clusters[j]]['Latitude'].mean(), df[df['Cluster'] == clusters[j]]['Longitude'].mean()),
                    unit=Unit.KILOMETERS)
        # track progress
    t2 = time.time()
    print(f"Time taken: {t2 -t1:0.2f} seconds")
    # save 
    np.save('data/distance_matrix.npy', distance_matrix)
    return distance_matrix

if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    print("Computing distance matrix...")
    compute_distance_matrix(df)