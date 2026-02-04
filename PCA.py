import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ===============================
# STEP 1: Create simple dataset
# ===============================
data = {
    'Height': [150, 160, 170, 180, 190],
    'Weight': [50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# ===============================
# STEP 2: Standardize the data
# ===============================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# ===============================
# STEP 3: Apply PCA
# ===============================
pca = PCA(n_components=1)   # reduce to 1 component
principal_components = pca.fit_transform(scaled_data)

# ===============================
# STEP 4: Display results
# ===============================
df_pca = pd.DataFrame(principal_components, columns=['Principal Component 1'])

print("\nPCA Output:\n")
print(df_pca)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
