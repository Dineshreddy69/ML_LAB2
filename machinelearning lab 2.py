# first question  calculating the euclidian distance and manhatandistance

import math
def euclidiandistance(A,B):
    distance=0
    for i in range(len(A)):
        distance+= (A[i]-B[i])**2
    return math.sqrt(distance)
   
pointA = [2,4,7]
pointB = [2,3,4]
print("\n the distance between two objects using euclidiandistance is :",euclidiandistance(pointA, pointB))

def manhatandistance(A,B):
    distance=0
    for i in range(len(A)):
        distance=distance+abs(A[i]-B[i])
    return distance

a=(3,4)
b=(5,6)
c=(7,8)
print(" the manhatan distance from A to B:",{manhatandistance(a,b)})
print(" the Manhatan distance from A to C :",{manhatandistance(a,c)})

# second question 
import numpy as np

def euclidean_distance(vec1, vec2):
   
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    #used zip function for iterating in the euclidean formula
    squared_diff = [(x - y) ** 2 for x, y in zip(vec1, vec2)]
    # apply square root for the formula
    euclidean_dist = np.sqrt(sum(squared_diff))
    
    return euclidean_dist


def k_nearest_neighbors(train_data, labels, query, k):
    
    distances = [euclidean_distance(query, data) for data in train_data]
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = [labels[i] for i in nearest_indices]
    predicted_label = max(set(nearest_labels), key=nearest_labels.count)
    
    return predicted_label


# data for  question 2

train_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
labels = ['A', 'B', 'A']
query_vec = [2, 3, 4]
k_value = 2
predicted_label = k_nearest_neighbors(train_data, labels, query_vec, k_value)
print("Predicted label:", predicted_label)

# question 3 
def label_encode_categorical(data):
    label_mapping = {}
    label_counter = 0
    
    for category in data:
        if category not in label_mapping:
            label_mapping[category] = label_counter
            label_counter += 1
    
    return label_mapping
if __name__=="__main__":
    categorical_data = ['red', 'blue', 'green', 'red', 'yellow', 'blue']
    label_mapping = label_encode_categorical(categorical_data)
    print("Label Mapping:", label_mapping)
    encoded_data = [label_mapping[category] for category in categorical_data]
    print("Encoded Data:", encoded_data)


    # question no 4 
    def one_hot_encode_categorical(data):
    unique_categories = list(set(data))
    unique_categories.sort()
    encoded_data = []
    for category in data:
        encoded_category = [0] * len(unique_categories)
        category_index = unique_categories.index(category)
        encoded_category[category_index] = 1
        encoded_data.append(encoded_category)
    return encoded_data

categorical_data = ['red', 'blue', 'green', 'red', 'yellow', 'blue']
encoded_data = one_hot_encode_categorical(categorical_data)
print("One-Hot Encoded Data:")
for data_point in encoded_data:
    print(data_point)
s