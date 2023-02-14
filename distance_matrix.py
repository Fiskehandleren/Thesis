from haversine import haversine, Unit
import numpy as np
import time
# compute the haversine distance between each station and all other stations
def compute_distance_matrix(df):
    t1 = time.time()
    stations = df['Station Name'].unique()
    distance_matrix = np.zeros((len(stations), len(stations)))
    print(len(stations))
    for i in range(len(stations)):
        for j in range(len(stations)):
            if i == j:
                continue
            elif distance_matrix[j][i] != 0:
                distance_matrix[i][j] = distance_matrix[j][i]
            else:
                distance_matrix[i][j] = haversine(
                    (df[df['Station Name'] == stations[i]]['Latitude'].values[0], df[df['Station Name'] == stations[i]]['Longitude'].values[0]), 
                    (df[df['Station Name'] == stations[j]]['Latitude'].values[0], df[df['Station Name'] == stations[j]]['Longitude'].values[0]),
                    unit=Unit.KILOMETERS)
        # track progress
        if i % 10 == 0:
            print(i)
    t2 = time.time()
    print(f"Time taken: {t2 -t1} seconds")
    return distance_matrix