from mhealth_activity import Recording
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

all_train_data_fn = []
for i in range(0, 280):
    all_train_data_fn.append(f'data/test/test_trace_{i:03d}.pkl')

print('Number of files:', len(all_train_data_fn))

data_list = []

for i, fn in enumerate(all_train_data_fn):
    rec = Recording(fn)
    timestamps_altitude = rec.data['altitude'].timestamps
    
    data_list.extend([
        {"id": i, "time": ts, "value": val}
        for ts, val in zip(timestamps_altitude, rec.data['altitude'].values)
    ])

print('loaded data')

df = pd.DataFrame(data_list)

extracted_features = extract_features(df, column_id='id', column_sort='time', impute_function=impute)

extracted_features.to_csv('extracted_features_test.csv')

print('saved extraced features to extracted_features_test.csv')