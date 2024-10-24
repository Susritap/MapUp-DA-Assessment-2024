#!/usr/bin/env python
# coding: utf-8

# 1.Write a function that takes a list and an integer n, and returns the list with every group of n elements reversed. If there are fewer than n elements left at the end, reverse all of them.
# 
# 

# In[2]:


# one approach

from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result_lst = []

    for i in range(0, len(lst), n):
        
        group = lst[i:i + n]
        result_lst.extend(group[::-1])
    
    return result_lst

# Example usage
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 2))  
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  
print(reverse_by_n_elements([10, 20, 30, 40, 50], 2))      
print(reverse_by_n_elements([1], 2))                        


# 2. Write a function that takes a list of strings and groups them by their length. The result should be a dictionary where:
# 
# 

# In[63]:


# one approach

from typing import Dict, List

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    result = {}

    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)

    return dict(sorted(result.items()))

# Example usage
strings = ["apple", "banana", "fig", "grape", "kiwi", "orange", "pear"]
grouped = group_by_length(strings)
print(grouped)


# 3. You are given a nested dictionary that contains various details (including lists and sub-dictionaries). Your task is to write a Python function that flattens the dictionary such that:
# 
# Nested keys are concatenated into a single key with levels separated by a dot (.).
# List elements should be referenced by their index, enclosed in square brackets (e.g., sections[0]).
# 

# In[64]:


from typing import Any, Dict

def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = {}
    
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_dict(item, f"{new_key}[{i}]", sep=sep))
                else:
                    items[f"{new_key}[{i}]"] = item
        else:
            items[new_key] = value
            
    return items

# Example usage
nested_dict = {
    'name': 'John',
    'age': 30,
    'address': {
        'street': '123 Main St',
        'city': 'Anytown'
    },
    'hobbies': ['reading', 'traveling'],
    'education': {
        'highschool': {
            'name': 'Anytown High',
            'graduation_year': 2010
        },
        'college': {
            'name': 'State University',
            'graduation_year': 2014
        }
    }
}

flattened = flatten_dict(nested_dict)
print(flattened)


# 4.) You are given a list of integers that may contain duplicates. Your task is to generate all unique permutations of the list. The output should not contain any duplicate permutations.
# 
# 

# In[67]:


from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])
            return
        
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] not in seen:
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]  # Swap
                backtrack(start + 1)  # Recurse
                nums[start], nums[i] = nums[i], nums[start]  # Backtrack

    result = []
    nums.sort() 
    backtrack(0)
    return result

nums = [1, 1, 2]
permutations = unique_permutations(nums)

for perm in permutations:
    print(perm)


# In[68]:


# another approach

from itertools import permutations

def unique_permutations(num):

    unique_perms = set(permutations(num))
    
    return [list(perm) for perm in unique_perms]

input_list = [1, 1, 2]
unique_perms = unique_permutations(input_list)

for perm in unique_perms:
    print(perm)


# 5. You are given a string that contains dates in various formats (such as "dd-mm-yyyy", "mm/dd/yyyy", "yyyy.mm.dd", etc.). Your task is to identify and return all the valid dates present in the string.
# 
# 

# In[69]:


import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b',
        r'\b(\d{2})/(\d{2})/(\d{4})\b',
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b'
    ]
    
    combined_pattern = '|'.join(patterns)
    matches = re.findall(combined_pattern, text)
    valid_dates = []

    for match in matches:
        if match[0]:
            valid_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3]:
            valid_dates.append(f"{match[3]}/{match[4]}/{match[5]}")
        elif match[6]:
            valid_dates.append(f"{match[6]}.{match[7]}.{match[8]}")

    return valid_dates

text = "Some dates are 12-05-2023, 05/12/2023, and 2023.05.12 in the text."
found_dates = find_all_dates(text)
print(found_dates)


# 6. Decode Polyline, Convert to DataFrame with Distances

# In[40]:


pip install polyline


# In[71]:


import pandas as pd
import polyline
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    coords = polyline.decode(polyline_str)
    
    data = {
        'latitude': [lat for lat, lon in coords],
        'longitude': [lon for lat, lon in coords],
        'distance': [0]  # Initialize with 0 for the first row
    }
    
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i - 1]
        lat2, lon2 = coords[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        data['distance'].append(distance)
    
    df = pd.DataFrame(data)
    return df

polyline_str = "u{~vHlmqM~aCl@qA|@~cBb@"
result_df = polyline_to_dataframe(polyline_str)
print(result_df)


# 7. Matrix Rotation and Transformation
# 

# In[3]:


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - i - 1] = matrix[i][j]

    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
print(result)


# 8. Time check

# In[6]:


import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    df['startTime'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.time
    df['endTime'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.time

    grouped = df.groupby(['id', 'id_2'])

    results = []
    
    for (id_val, id_2_val), group in grouped:
        days_covered = set(group['startDay']) | set(group['endDay'])
        all_days_present = len(days_covered) == 7
        
        times = []
        for _, row in group.iterrows():
            times.append((row['startTime'], row['endTime']))
        
        sorted_times = sorted(times)
        full_day_covered = sorted_times[0][0] == pd.to_datetime('00:00:00').time() and sorted_times[-1][1] == pd.to_datetime('23:59:59').time()
        
        is_complete = all_days_present and full_day_covered
        results.append(((id_val, id_2_val), not is_complete))
    
    index = pd.MultiIndex.from_tuples([r[0] for r in results], names=['id', 'id_2'])
    bool_series = pd.Series([r[1] for r in results], index=index)
    
    return bool_series


# In[8]:


df=pd.read_csv('dataset-1.csv')
result = time_check(df)
print(result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




