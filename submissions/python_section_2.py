#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 9.) Distance Matix Calculation

import pandas as pd
import numpy as np

def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    unique_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    for _, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] += row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] += row['distance']  # For symmetry

    for i in unique_ids:
        for j in unique_ids:
            if distance_matrix.loc[i, j] > 0:
                for k in unique_ids:
                    if distance_matrix.loc[j, k] > 0:
                        distance_matrix.loc[i, k] += distance_matrix.loc[i, j] + distance_matrix.loc[j, k]

    distance_matrix = distance_matrix.where(pd.DataFrame(np.triu(np.ones(distance_matrix.shape), k=1), 
                                  index=distance_matrix.index, columns=distance_matrix.columns) == 1, 
                                  distance_matrix).fillna(0)

    return distance_matrix


distance_df = calculate_distance_matrix('dataset-2.csv')
print(distance_df)


# In[4]:


# 10. Unroll Distance Matrix

import pandas as pd

def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    unrolled_data = []

    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude same id_start to id_end pairs
                distance = distance_matrix.loc[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    unrolled_df = pd.DataFrame(unrolled_data)
    return unrolled_df

# Example usage with the distance matrix from Question 9
distance_df = calculate_distance_matrix('dataset-2.csv')  # Assuming this function has been defined
unrolled_df = unroll_distance_matrix(distance_df)
print(unrolled_df)


# In[11]:


# 11.) Finding ID's with percentage Threshold

def find_ids_within_ten_percentage_threshold(unrolled_df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    reference_avg_distance = unrolled_df[unrolled_df['id_start'] == reference_id]['distance'].mean()
    lower_bound = reference_avg_distance * 0.90
    upper_bound = reference_avg_distance * 1.10
    avg_distances = unrolled_df.groupby('id_start')['distance'].mean()
    ids_within_threshold = avg_distances[(avg_distances >= lower_bound) & (avg_distances <= upper_bound)].index
    return pd.DataFrame({'id_start': ids_within_threshold}).sort_values(by='id_start')

# Example usage
avg_distances = unrolled_df.groupby('id_start')['distance'].mean()
reference_id = avg_distances.idxmax()  # This will select the ID with the maximum average distance
result_threshold_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print(result_threshold_df)


# In[12]:


import pandas as pd

def calculate_toll_rate(unrolled_df: pd.DataFrame) -> pd.DataFrame:
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    unrolled_df['moto'] = unrolled_df['distance'] * rate_coefficients['moto']
    unrolled_df['car'] = unrolled_df['distance'] * rate_coefficients['car']
    unrolled_df['rv'] = unrolled_df['distance'] * rate_coefficients['rv']
    unrolled_df['bus'] = unrolled_df['distance'] * rate_coefficients['bus']
    unrolled_df['truck'] = unrolled_df['distance'] * rate_coefficients['truck']

    return unrolled_df

# Example usage
toll_rate_df = calculate_toll_rate(unrolled_df)
print(toll_rate_df)


# In[14]:


import datetime

def calculate_time_based_toll_rates(unrolled_df: pd.DataFrame) -> pd.DataFrame:
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_ranges = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0)),  
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0)), 
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59)) 
    ]

    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    result_data = []
    
    for _, row in unrolled_df.iterrows():
        for day in days_of_week:
            is_weekend = day in ['Saturday', 'Sunday']
            for i, (start_time, end_time) in enumerate(time_ranges):
                toll_row = row.copy()
                toll_row['start_day'] = day
                toll_row['end_day'] = day
                toll_row['start_time'] = start_time
                toll_row['end_time'] = end_time

                if is_weekend:
                    factor = weekend_discount_factor
                else:
                    factor = weekday_discount_factors[i]

                toll_row['moto'] *= factor
                toll_row['car'] *= factor
                toll_row['rv'] *= factor
                toll_row['bus'] *= factor
                toll_row['truck'] *= factor
                
                result_data.append(toll_row)

    final_df = pd.DataFrame(result_data)
    return final_df

time_based_df = calculate_time_based_toll_rates(unrolled_df)
print(time_based_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




