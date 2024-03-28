import pandas as pd
import sklearn.datasets
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv('/Users/ruoshui/Desktop/MKT512/IA2/crews-cup.csv')
print(df.head())
print(df.isnull().any())
#Correlation
dfcorr=df.drop(['id','csat','intent','origination','classes','profit'],axis=1)
print(dfcorr.head())
df_corrresult=dfcorr.corr()
print(df_corrresult)
# correlation heatmap
plt.figure(figsize=(8, 6), facecolor='w')
ax = sns.heatmap(df_corrresult, square=True, annot=True, fmt='.3f', 
                 linewidth=1, cmap='coolwarm',linecolor='white', cbar=True,
                 annot_kws={'size':9,'weight':'normal','color':'white'},
                 cbar_kws={'fraction':0.046, 'pad':0.03}) 
plt.savefig("heatmap_correlation.png", dpi=600)
plt.show()
#How many factors do I use-Scree plot
fa=FactorAnalyzer(n_factors=dfcorr.shape[1],rotation = None)
fa.fit(dfcorr)
    # get eigen values
ev, _ = fa.get_eigenvalues()
    # Scree Plot
plt.scatter(range(1, dfcorr.shape[1]+1), ev)
plt.plot(range(1, dfcorr.shape[1]+1), ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
#EV and variance
ev, v = fa.get_eigenvalues()
print('Eigenvalues:', ev)
print('Variance explained:', fa.get_factor_variance())
#EFA
fa = FactorAnalyzer(n_factors=5, rotation=None)
fa.fit(dfcorr)
# get loading factors
loadings = fa.loadings_
print(loadings)

fa_varimax = FactorAnalyzer(n_factors=5, rotation='varimax')
fa_varimax.fit(dfcorr)
loadings_varimax = fa_varimax.loadings_
print(loadings_varimax)

# get factor score  
factor_scores = fa.transform(dfcorr)
corr_matrix = pd.DataFrame(factor_scores).corr()

# get heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Factor Correlation')
plt.show()

#Validate
import pingouin as pg
factor1 = dfcorr.iloc[:, 0:3]
factor2 = dfcorr.iloc[:, 3:6]
factor3 = dfcorr.iloc[:, 6:9]
factor4 = dfcorr.iloc[:, 9:12]
factor5 = dfcorr.iloc[:, 12:15]
# calculate Cronbach's alpha
cronbach_alpha = pg.cronbach_alpha(data=factor1)
print("Cronbach's alpha for Factor 1:", cronbach_alpha[0])
print("95% CI for Factor 1:", cronbach_alpha[1])
alphas_dropped = {}
for attribute in factor1.columns:
    temp_df = factor1.drop(attribute, axis=1)
    alphas_dropped[attribute] = pg.cronbach_alpha(data=temp_df)[0]

# output result
for attribute, alpha in alphas_dropped.items():
    print(f"Cronbach's alpha if '{attribute}' is dropped: {alpha}")
#Factor2
cronbach_alpha = pg.cronbach_alpha(data=factor2)
print("Cronbach's alpha for Factor 2:", cronbach_alpha[0])
print("95% CI for Factor 2:", cronbach_alpha[1])
alphas_dropped = {}
for attribute in factor2.columns:
    temp_df = factor2.drop(attribute, axis=1)
    alphas_dropped[attribute] = pg.cronbach_alpha(data=temp_df)[0]

# output result
for attribute, alpha in alphas_dropped.items():
    print(f"Cronbach's alpha if '{attribute}' is dropped: {alpha}")

 #Factor3
cronbach_alpha = pg.cronbach_alpha(data=factor3)
print("Cronbach's alpha for Factor 3:", cronbach_alpha[0])
print("95% CI for Factor 3:", cronbach_alpha[1])
alphas_dropped = {}
for attribute in factor3.columns:
    temp_df = factor3.drop(attribute, axis=1)
    alphas_dropped[attribute] = pg.cronbach_alpha(data=temp_df)[0]

# output result
for attribute, alpha in alphas_dropped.items():
    print(f"Cronbach's alpha if '{attribute}' is dropped: {alpha}")

#Factor4
cronbach_alpha = pg.cronbach_alpha(data=factor4)
print("Cronbach's alpha for Factor 4:", cronbach_alpha[0])
print("95% CI for Factor 4:", cronbach_alpha[1])
alphas_dropped = {}
for attribute in factor4.columns:
    temp_df = factor4.drop(attribute, axis=1)
    alphas_dropped[attribute] = pg.cronbach_alpha(data=temp_df)[0]

# output result
for attribute, alpha in alphas_dropped.items():
    print(f"Cronbach's alpha if '{attribute}' is dropped: {alpha}")

#Factor5
cronbach_alpha = pg.cronbach_alpha(data=factor5)
print("Cronbach's alpha for Factor 5:", cronbach_alpha[0])
print("95% CI for Factor 5:", cronbach_alpha[1])
alphas_dropped = {}
for attribute in factor5.columns:
    temp_df = factor5.drop(attribute, axis=1)
    alphas_dropped[attribute] = pg.cronbach_alpha(data=temp_df)[0]

# output result
for attribute, alpha in alphas_dropped.items():
    print(f"Cronbach's alpha if '{attribute}' is dropped: {alpha}")

#K-means
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
df_kmeans=df.drop(['id'],axis=1)
#standardize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_kmeans)
wcss = []
# try different cluster number from 1-10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) 
plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within-cluster sum of squares
plt.show()
#k-means analysis
k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X_scaled)

labels = kmeans.labels_

# segmentation analysis
df['Cluster'] = labels
print(df.head())
df_segment_0 = df[df['Cluster'] == 0]
df_segment_1 = df[df['Cluster'] == 1]
df_segment_2 = df[df['Cluster'] == 2]

num_segment_0 = df_segment_0.shape[0]
num_segment_1 = df_segment_1.shape[0]
num_segment_2 = df_segment_2.shape[0]

print("Number of customers in Segment 0:", num_segment_0)
print("Number of customers in Segment 1:", num_segment_1)
print("Number of customers in Segment 2:", num_segment_2)

summary_segment_0 = df_segment_0.describe()
summary_segment_1 = df_segment_1.describe()
summary_segment_2 = df_segment_2.describe()

print("Segment 0 Summary:\n", summary_segment_0)
print("\nSegment 1 Summary:\n", summary_segment_1)
print("\nSegment 2 Summary:\n", summary_segment_2)

writer = pd.ExcelWriter('~/Desktop/clustering_result1.xlsx', engine='xlsxwriter')


summary_segment_0.to_excel(writer, sheet_name='Segment 0 Summary')
summary_segment_1.to_excel(writer, sheet_name='Segment 1 Summary')
summary_segment_2.to_excel(writer, sheet_name='Segment 2 Summary')

# save Excel
writer.save()
df['Avg_Forced']= df[['f1', 'f2', 'f3']].mean(axis=1)
df['Avg_Habitual']= df[['h1', 'h2', 'h3']].mean(axis=1)
df['Avg_Economic']= df[['e1', 'e2', 'e3']].mean(axis=1)
df['Avg_Affirmative']= df[['a1', 'a2', 'a3']].mean(axis=1)
df['Avg_Normative']= df[['n1', 'n2', 'n3']].mean(axis=1)

print(df.head())
print(df.describe())

import statsmodels.api as sm
from statsmodels.formula.api import ols

# ANOVA test for each segment
anova_results = {}
Avg_scores = ['Avg_Forced', 'Avg_Habitual', 'Avg_Economic', 'Avg_Affirmative', 'Avg_Normative']
for score in Avg_scores:
    model = ols(f'{score} ~ C(Cluster)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_results[score] = anova_table

# ANOVA result
for score, result in anova_results.items():
    print(f'ANOVA result for {score}:\n{result}\n')

# visualization
plt.figure(figsize=(10, 8))
for i, score in enumerate(Avg_scores, 1):
    plt.subplot(3, 2, i)  
    sns.boxplot(x='Cluster', y=score, data=df)
    plt.title(f'Scores for {score} by Cluster')

plt.tight_layout()
plt.show()

df.to_excel('~/Desktop/regression.xlsx', index=False)


