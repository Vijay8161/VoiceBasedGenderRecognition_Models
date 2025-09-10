import numpy as np

# Option 2 (simplest fix)
data = np.load("data/preprocessed/features_batch_1751000857.npz")

print(data.files)            # Should show ['X', 'y']
print(data['X'].shape)       # Should show (N, D)
print(data['y'].shape)       # Should match N
