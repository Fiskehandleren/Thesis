import multiprocessing
import numpy as np
import time
import pandas as pd 
from haversine import haversine, Unit


def compute_distance(i, stations, distance_matrix, df):
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

def create_chunks(num_stations):
    chunk_size = num_stations // multiprocessing.cpu_count()
    chunks = []
    for i in range(0, num_stations, chunk_size):
        if i + chunk_size > num_stations:
            chunks.append(range(i, num_stations))
        else:
            chunks.append(range(i, i + chunk_size))
    return chunks

def compute_distance_matrix(df):
    t1 = time.time()
    stations = df['Station Name'].unique()
    distance_matrix = np.zeros((len(stations), len(stations)))
    chunks = create_chunks(len(stations))
    print(chunks)
    processes = [multiprocessing.Process(target=compute_distance, args=(i, stations, distance_matrix, df)) for chunk in chunks for i in chunk]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    t2 = time.time()
    print("Time taken: ", t2-t1)
    return distance_matrix





if __name__ == '__main__':
    df = pd.read_csv('ChargePoint Data CY20Q4.csv', dtype={'Station Name': str})

    distance_matrix = compute_distance_matrix(df)

