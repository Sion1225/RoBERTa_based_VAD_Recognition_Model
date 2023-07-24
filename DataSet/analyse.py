import matplotlib.pyplot as plt
from scipy.stats import kstest, norm
import pandas as pd

# Load the data
data = pd.read_csv("C:\\Users\\Siwon\\Documents\\GitHub\\Assinging_VAD_scores_BERT\\DataSet\\emobank.csv")

# Create histograms for V, A, D
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Variables to check
variables = ['V', 'A', 'D']

# Test each variable
for var, ax in zip(variables, axes):
    # Histogram
    data[var].hist(ax=ax)
    ax.set_title(f'Histogram of {var}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    # Kolmogorov-Smirnov test
    # Calculate the mean and standard deviation of the data
    mean, std = data[var].mean(), data[var].std()
    
    # Standardize the data
    standardized_data = (data[var] - mean) / std

    # Kolmogorov-Smirnov test
    stat, p = kstest(standardized_data, 'norm')
    print(f'\nKolmogorov-Smirnov Test for {var}:')
    print(f'Statistic={stat}, p={p}')
    if p > 0.05:
        print(f'The data may be considered normally distributed (fail to reject H0)\n')
    else:
        print(f'The data may not be considered normally distributed (reject H0)\n')

plt.tight_layout()
plt.show()
