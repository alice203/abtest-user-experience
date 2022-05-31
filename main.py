import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from scipy import stats
from scipy.stats import norm, chi2
from statsmodels.stats.weightstats import ztest as ztest

########### ---- Load data
df_assignment = pd.read_csv('data/df_assignment.csv', parse_dates=['assignment_date', 'install_date']).iloc[:,1:]
df_activity = pd.read_csv('data/df_activity.csv').iloc[:,1:]
df_activity_before = pd.read_csv('data/df_activity_before.csv').iloc[:,1:]
conv_over_time = pd.read_csv('data/conv_over_time.csv', parse_dates=['date']).iloc[:,1:]
activity_over_time = pd.read_csv('data/activity_over_time.csv', parse_dates=['date']).iloc[:,1:]

# Merge player datasets
pdata = pd.merge(df_assignment,df_activity,on='playerid',how='inner')
pdata_before = pd.merge(df_assignment,df_activity_before,on='playerid',how='inner')

# Merge time datasets and create new variables
tdata = (pd.merge(activity_over_time, conv_over_time, 
                  on=['date','abtest_group'],how='inner')).sort_values(by=['date', 'abtest_group'])
tdata['cvr'] = tdata['number_of_conversions']/tdata['number_of_players']
tdata['avg_purchase'] = tdata['sum_purchase']/tdata['number_of_players']


########### ---- EDA
#pdata.describe()
#pdata.isnull().sum()
total_players_A = pdata[pdata['abtest_group']=='A']['playerid'].count()
total_players_B = pdata[pdata['abtest_group']=='B']['playerid'].count()
total_players_all = pdata['playerid'].count()
#print('Number of unique players:', pdata['playerid'].count())
#print('Number of players in control group A:', total_players_A)
#print('Number of players in treatment group B:', total_players_B)
#print('Group A (in percentage): ', (total_players_A/total_players_all)*100)
#print('Group B (in percentage): ', (total_players_B/total_players_all)*100)
#print('Number of new players in experiment phase:', pdata[pdata['install_date']>'2017-05-04']['playerid'].count())

########### ---- Sanity check pre-experiment
# Check group sizes
dem_summary = pdata_before.pivot_table(values='times_played', index='abtest_group', aggfunc=np.mean)
dem_summary['avg_gamesend'] = pdata_before.pivot_table(values='avg_gameends', index='abtest_group', aggfunc=np.mean)
dem_summary['avg_purchase'] = pdata_before.pivot_table(values='avg_purchase', index='abtest_group', aggfunc=np.mean)
dem_summary['total'] = pdata.pivot_table(values='times_played', index='abtest_group', aggfunc=lambda x: len(x))
#dem_summary

# Pie chart: Player assignment
labels = 'A', 'B'
sizes = [80.01, 19.99]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  
plt.show()


# Pre-experiment test
activity_over_time.head()

df_group = pdata_before.groupby(['abtest_group']).mean()
df_group_std = pdata_before.groupby(['abtest_group']).std()
avg_gamerounds= df_group['avg_gameends']
std_gamerounds= df_group_std['avg_gameends']

gamerounds = pd.DataFrame(avg_gamerounds)
gamerounds['std_gamerounds']=std_gamerounds
gamerounds = gamerounds.reset_index(drop=False)
gamerounds

#Z-test: H-0: The average gamesround per player is the same for the two groups
mu_B = gamerounds.iloc[1,1]
mu_A = gamerounds.iloc[0,1]
std_B = gamerounds.iloc[1,2]
std_A = gamerounds.iloc[0,2]
n_B = total_players_B
n_A = total_players_A
Z = (mu_A - mu_B)/np.sqrt(std_A**2/n_A + std_B**2/n_B)
pvalue = norm.sf(Z)
print("Z-score: {0}\np-value: {1}".format(Z,pvalue))
α = 0.01
print('p-value < α: ', pvalue<α)

#EDIT: Using statsmodels library
groupA = pdata_before.query("abtest_group=='A'")['avg_gameends']
groupB = pdata_before.query("abtest_group=='B'")['avg_gameends']

ztest(groupA, groupB, value=0) 


########### ---- Post-experiment analysis
df_group = pdata.groupby(['abtest_group']).mean()
df_group_std = pdata.groupby(['abtest_group']).std()
avg_gamerounds= df_group['avg_gameends']
std_gamerounds= df_group_std['avg_gameends']

gamerounds = pd.DataFrame(avg_gamerounds)
gamerounds['std_gamerounds']=std_gamerounds
gamerounds = gamerounds.reset_index(drop=False)

#Z-test: H-0: The average gamesround per player is the same for the two groups
mu_B = gamerounds.iloc[1,1]
mu_A = gamerounds.iloc[0,1]
std_B = gamerounds.iloc[1,2]
std_A = gamerounds.iloc[0,2]
n_B = total_players_B
n_A = total_players_A
Z = (mu_A - mu_B)/np.sqrt(std_A**2/n_A + std_B**2/n_B)
pvalue = norm.sf(Z)
print("Z-score: {0}\np-value: {1}".format(Z,pvalue))
α = 0.01
print('p-value < α: ', pvalue<α)

#Trimmed data
data_clean_995 = pdata[pdata['avg_gameends']<pdata['avg_gameends'].quantile(0.995)] #0.5%
data_clean_99 = pdata[pdata['avg_gameends']<pdata['avg_gameends'].quantile(0.99)] #1.0%
data_clean_975 = pdata[pdata['avg_gameends']<pdata['avg_gameends'].quantile(0.975)] #2.5%
df_group = data_clean_975.groupby(['abtest_group']).mean()
df_group_std = data_clean_975.groupby(['abtest_group']).std()
avg_gamerounds1= df_group['avg_gameends']
std_gamerounds1= df_group_std['avg_gameends']

#Z-test with trimmed data
mu_B = avg_gamerounds1[1]
mu_A = avg_gamerounds1[0]
std_B = std_gamerounds1[1]
std_A = std_gamerounds1[0]
n_B = total_players_B
n_A = total_players_A
Z = (mu_A - mu_B)/np.sqrt(std_A**2/n_A + std_B**2/n_B)
pvalue = norm.sf(Z)
print("Z-score: {0}\np-value: {1}".format(Z,pvalue))

ax = sns.violinplot(x="abtest_group", y="avg_gameends", data=data_clean_975)
ax.set_title('Distribution of avg. game rounds per player per group (*2.5% right-trimmed data) ')
ax.set_ylabel('Average game rounds')
ax.set_xlabel('AB test group')
ax.figure.savefig("plot3.png")

# ### Purchases
sum_purchase = pdata.groupby(['abtest_group']).sum()
std_purchase = pdata.groupby(['abtest_group']).std()
sum_purchase= df_group['avg_purchase']
std_purchase= df_group_std['avg_purchase']

total_purchase = pd.DataFrame(sum_purchase)
total_purchase['std_purchase'] = std_purchase
total_purchase = total_purchase.reset_index(drop=False)

# Average spend per player per groups
avg_purchase_A = total_purchase.iloc[0,1]/total_players_A
avg_purchase_B = total_purchase.iloc[1,1]/total_players_B
print('Average purchase per player group A: ', round(avg_purchase_A , 4))
print('Average purchase per player group B: ', round(avg_purchase_B,4))

# Z-test: H-0: The average amount of purchase per player is the same for the two groups
mu_B = total_purchase.iloc[1,1]
mu_A = total_purchase.iloc[0,1]
std_B = total_purchase.iloc[1,2]
std_A = total_purchase.iloc[0,2]
n_B = total_players_B
n_A = total_players_A
Z = (mu_B- mu_A)/np.sqrt(std_B**2/n_B + std_A**2/n_A)
pvalue = norm.sf(Z)
print("Z-score: {0}\np-value: {1}".format(Z,pvalue))
print('p-value < α: ', pvalue<α)

# Trimmed data
#print(pdata['avg_purchase'].quantile(0.995))
#print(pdata['avg_purchase'].quantile(0.99))
#print(pdata['avg_purchase'].quantile(0.975))
data_clean_995 = pdata[pdata['avg_purchase']<pdata['avg_purchase'].quantile(0.995)] #0.5%
data_clean_99 = pdata[pdata['avg_purchase']<pdata['avg_purchase'].quantile(0.99)] #1%
data_clean_975 = pdata[pdata['avg_purchase']<pdata['avg_purchase'].quantile(0.975)] #2.5%
df_group = data_clean_975.groupby(['abtest_group']).mean()
df_group_std = data_clean_975.groupby(['abtest_group']).std()
avg_purchase1= df_group['avg_purchase']
std_purchase1= df_group_std['avg_purchase']
#Z-test with trimmed data
mu_B = avg_purchase1[1]
mu_A = avg_purchase1[0]
std_B = std_purchase1[1]
std_A = std_purchase1[0]
n_B = total_players_B
n_A = total_players_A
Z = (mu_B- mu_A)/np.sqrt(std_B**2/n_B + std_A**2/n_A)
pvalue = norm.sf(Z)
print("Z-score: {0}\np-value: {1}".format(Z,pvalue))
print('p-value < α: ', pvalue<α)


# # Difference in conversions between groups
#Add a conversion variable
pdata['conversion'] = np.where((pdata['conversion_date'].notna())&(pdata['conversion_date']>='2017-05-04'), 1, 0)

#Conversion rate 
total_converts_A = pdata[pdata['abtest_group']=='A']['conversion'].sum()
total_converts_B = pdata[pdata['abtest_group']=='B']['conversion'].sum()
total_converts_all = pdata['conversion'].sum()
cvr_A = total_converts_A/total_players_A
cvr_B = total_converts_B/total_players_B

# Model H-0: Players in group A and B are equally converting
cvr_all = total_converts_all/total_players_all

# Summary table for conversions
conversion_data = pdata[['abtest_group', 'conversion', 'playerid']]
ab_summary = conversion_data.pivot_table(values='conversion', index='abtest_group', aggfunc=np.sum)
# add additional columns to the pivot table
ab_summary['total'] = conversion_data.pivot_table(values='conversion', index='abtest_group', aggfunc=lambda x: len(x))
ab_summary['rate'] = conversion_data.pivot_table(values='conversion', index='abtest_group')
ab_summary['std'] = conversion_data.pivot_table(values='conversion', index='abtest_group', aggfunc=np.std)
ab_summary['non-conversions'] = ab_summary['total'] - ab_summary['conversion']

# Theoretical outcome based on H-0
TA_1 = round(cvr_all*ab_summary.iloc[0,1])
TB_1 = round(cvr_all*ab_summary.iloc[1,1])
TA_2 = total_players_A - TA_1
TB_2 = total_players_B - TB_1

T = np.array([TA_1, TB_1, TA_2, TB_2])
O = np.array([ab_summary.iloc[0,0], ab_summary.iloc[1,0], ab_summary.iloc[0,3],ab_summary.iloc[1,3]])
D = np.sum(np.square(T-O)/T)
pvalue = chi2.sf(D, df=1)
print("distance d: {0}\np-value: {1}".format(D,pvalue))
print('p-value < α: ', pvalue<α)

fig1 = px.line(tdata, x='date', y='cvr', color='abtest_group')
fig1.add_shape(dict(type="line", x0="2017-05-04", y0=0.0001, x1="2017-05-04", y1=0.0005, line_dash="dot", 
                   line=dict(color="orange", width=3)))
fig1.update_layout(title="Conversion rate before and during treatment per group", xaxis_title="Date in time", 
                  yaxis_title="Conversion rate")
fig1

# # Reactions from different players
#Add 'purchaser' variable to detect trends for users who purchase
pdata['purchaser'] = np.where(pdata['sum_purchase']==0, 0, 1)
#Add 'new-player' variable to detect trends for new players installing the game in the treatment phase
pdata['new_player'] = np.where((pdata['install_date']>='2017-05-04'), 1, 0)
pdata_A = pdata[pdata['abtest_group']=='A']
pdata_B = pdata[pdata['abtest_group']=='B']
#Testing whether the average round of games differs between new and old players for treatment group B
new_player_b = pdata_B.pivot_table(values='avg_gameends', index='new_player', aggfunc=np.mean)
new_player_b['std'] = pdata_B.pivot_table(values='avg_gameends', index='new_player', aggfunc=np.std)
print(new_player_b)
new_player_a = pdata_A.pivot_table(values='avg_gameends', index='new_player', aggfunc=np.mean)
new_player_a['std'] = pdata_A.pivot_table(values='avg_gameends', index='new_player', aggfunc=np.std)
#print(new_player_a)

#Z-test: H-0: The average gamesround per player is the same for new and old players in group B
mu_0 = new_player_b['avg_gameends'][0]
mu_1 = new_player_b['avg_gameends'][1]
std_0 = new_player_b['std'][0]
std_1 = new_player_b['std'][1]
n_0 = total_players_B
n_1 = total_players_B
Z = (mu_1 - mu_0)/np.sqrt(std_1**2/n_1 + std_0 **2/n_0)
pvalue = norm.sf(Z)
print("Z-score: {0}\np-value: {1}".format(Z,pvalue))
α = 0.01
print('p-value < α: ', pvalue<α)

#Z-test: H-0: The average gamesround per player is the same for new and old players in group A
mu_0 = new_player_a['avg_gameends'][0]
mu_1 = new_player_a['avg_gameends'][1]
std_0 = new_player_a['std'][0]
std_1 = new_player_a['std'][1]
n_0 = total_players_A
n_1 = total_players_A
Z = (mu_1 - mu_0)/np.sqrt(std_1**2/n_1 + std_0 **2/n_0)
pvalue = norm.sf(Z)
print("Z-score: {0}\np-value: {1}".format(Z,pvalue))
α = 0.01
print('p-value < α: ', pvalue<α)

ax1 = sns.violinplot(x="new_player", y="avg_gameends", data=pdata_B)
ax1.set_title('Distribution of avg. game rounds for new and old players in group B')
ax1.set_ylabel('Avg. game rounds')
ax1.set_xlabel('New player')

