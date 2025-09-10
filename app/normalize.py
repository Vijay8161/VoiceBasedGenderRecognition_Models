import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the NPZ files
data1 = np.load('data/preprocessed/augmented_features_batch_1751092828.npz')
data2 = np.load('data/preprocessed/features_batch_1751092828.npz')



# Assuming both files have arrays named 'X' and 'y'
array1_X = data1['X']
array1_y = data1['y']
array2_X = data2['X']
array2_y = data2['y']

# Combine data to fit the StandardScaler
combined_data = np.concatenate((array1_X, array2_X))
print("Combined data shape:", combined_data.shape)

# Initialize and fit the scaler
scaler = StandardScaler()
scaler.fit(combined_data)

# Normalize both datasets, preserving their original shapes
normalized_array1_X = scaler.transform(array1_X)
normalized_array2_X = scaler.transform(array2_X)

# Save the normalized data back to NPZ files, maintaining original structures
np.savez('normalized_aug.npz', X=normalized_array1_X, y=array1_y)
np.savez('normalized_reg.npz', X=normalized_array2_X, y=array2_y)

# Save the fitted scaler
joblib.dump(scaler, 'standard_scaler.pkl')

print("Normalization complete and data saved. StandardScaler stored.")
