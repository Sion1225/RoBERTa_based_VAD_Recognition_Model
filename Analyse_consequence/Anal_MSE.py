from scipy.stats import jarque_bera, shapiro
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

'''
# Ver 1.3
mse_values = (
    0.06595173478126526,
    0.07482276856899261,
    0.0639118105173111,
    0.0636352077126503,
    0.06066885590553284,
    0.06362489610910416,
    0.06730588525533676,
    0.06427466124296188,
    0.06829468160867691,
    0.06347636133432388,
    0.06701111793518066,
    0.07990781217813492,
    0.06524074822664261,
    0.05835770070552826,
    0.06026444584131241,
    0.0600387342274189,
    0.07191567122936249,
    0.0682518258690834,
    0.07462234795093536,
    0.07440381497144699,
    0.06417826563119888,
    0.0556042455136776,
    0.06735994666814804,
    0.07727619260549545,
    0.0672365128993988,
    0.06561418622732162,
    0.06592539697885513,
    0.06328296661376953,
    0.07598831504583359,
    0.05550612881779671
)
'''

# Ver 2.2
mse_values = (
    0.06225941702723503, 
    0.0721583440899849, 
    0.0683281198143959, 
    0.062137383967638016, 
    0.06356355547904968, 
    0.06152718886733055, 
    0.06778662651777267, 
    0.06201140210032463, 
    0.0674576386809349, 
    0.062480755150318146, 
    0.06353543698787689, 
    0.07306861132383347, 
    0.06848742812871933, 
    0.05941467732191086, 
    0.07424312084913254, 
    0.06297231465578079, 
    0.06942103803157806, 
    0.06933040916919708, 
    0.073894202709198, 
    0.07222463190555573, 
    0.0648893192410469, 
    0.05919027701020241, 
    0.06424389779567719, 
    0.06940481811761856, 
    0.07207347452640533, 
    0.0653950497508049, 
    0.06542855501174927, 
    0.06374996900558472, 
    0.07849754393100739, 
    0.05818257853388786
)

# Perform Jarque-Bera test
jb_stats = jarque_bera(mse_values)
print(f"Jarque-Bera stats\n{jb_stats}")

# Perform Shapiro-Wilk test
sw_stats = shapiro(mse_values)
print(f"Shapiro-Wilk stats\n{sw_stats}")


# Plot histogram of the data
plt.hist(mse_values, bins="auto", color="orange", edgecolor='black')
plt.title("Histogram of Model_s' MSE values")
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.show()

# Plot QQ-plot of the data
stats.probplot(mse_values, dist="norm", plot=plt)
plt.title("QQ-plot of Model_s' MSE values")
plt.show()


# Calculate sample mean and standard error
sample_mean = np.mean(mse_values)
standard_error = np.std(mse_values, ddof=1) / np.sqrt(len(mse_values))

# Calculate confidence interval
confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)
confidence_interval = [sample_mean - z_score * standard_error, sample_mean + z_score * standard_error]

print(f"Confidence Interval:\n{confidence_interval}")