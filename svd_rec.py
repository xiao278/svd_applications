import surprise
import numpy as np
import pandas as pd

reader = surprise.Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))
# Load the data into a Surprise dataset
data = surprise.Dataset.load_builtin('ml-100k').build_full_trainset()
data_list = []
for (user, item, rating) in data.all_ratings():
    data_list.append([user, item, rating])

df = pd.DataFrame(data_list)
df.columns = ['user', 'item', 'rating']
print(df)