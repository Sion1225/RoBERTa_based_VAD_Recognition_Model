from sklearn.utils import resample
import numpy as np

datas = (1,
    2,
    3,
    4,
    5) # Edit datas
samples = []
means = []

for i in range(10000):
    # Extract samples
    samples.append(resample(datas, replace=True, n_samples=len(datas)))

    mean = np.mean(samples)
    means.append(mean)

    with open("C:\\Users\\Siwon\\Documents\\GitHub\\Assinging_VAD_scores_BERT\\Analyse_consequence\\Bootstrap_Output.txt",'a') as f:
        f.write(f"{mean}\n")