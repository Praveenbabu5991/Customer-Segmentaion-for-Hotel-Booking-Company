# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 00:28:13 2018

@author: Bolt
"""
'''
CEO OF YPEDIA
1.A REPORT OF UNDERPERFORMING AND OVERPERFORMING SEGMENTS
2.HOW TO TAILOR THE NEW MARKETING CAMPAIGN FOR NEXT MONTH?
3.HOW TO IMPROVE THE USER BOOKING RATE?
WORK PLAN:
1.EXPLORE THE DATA 
2.TWO SAMPLE Z TEST 
-a report for overperforming and underperforming segment
3.K-MEANS CLUSTERING
-what cities to launch the marketing campaign next month?
4.DECISION TREE 
-how to improve the user booking rate 



'''
from os import chdir,getcwd
getcwd()
chdir('D:\mlproject\customersegmentation')


import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing 
from sklearn.cluster import KMeans
from sklearn import decomposition
import matplotlib.pyplot as plt


sample=pd.read_csv('sample.csv')


#business logic validation
#check in date >booking date
sample[sample['srch_ci']<sample['date_time']]
#42690*25 does not follow the business logic
#check out date>check in date
#no of guest >0
#more checks



#####################explore the data - descriptive statistics#####################
sample.head()
sample['user_location_city'].nunique()
sample.describe()
sample['date_time'].min()
sample['date_time'].max()
sample.info()              #checking null va;ues
sample.describe()
sample.dtypes
sample.isnull.sum()
sample.isnull.count()
# unique counts
def unique_counts(sample):
   for i in sample.columns:                #traversing through columns
       count = sample[i].nunique()
       print(i, ": ", count)
unique_counts(sample)

# correlations
sample['is_booking'].mean()
pd.crosstab(sample['is_booking'], sample['srch_rm_cnt'])
sample.groupby('srch_rm_cnt')['is_booking'].mean()
sample['srch_children_cnt'].corr(sample['is_booking'])
sample.corr()

# sample.hist()
sample[['channel', 'is_booking', 'is_mobile', 'orig_destination_distance', 'srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt']].hist()





'''
#selecting rows   
print(sample.iloc[0 ])
#subsetting rows and columns
print(sample.ix[[0,99,999],[0,3,5]])
#group and aggregation
sample.groupby('user_id').is_booking.count()
sample.user_id.mean()
sample.user_id.min()
sample.user_id.max()
sample.user_id.median()

sample.user_id.describe()

#replace
data['is_employed'].replace(to_replace=1,value=True,inplace=True,method=None)

#rename
sample.rename(columns={'occupation':'title'},inplace=True)
#sorting
sample.sort_values(by='col name',ascending=0)
#multiple column
sample.sort_value(by=['',''])

#concat
concat=pd.concat([df1,df2,df3])=join three tables with same columns
#merge
merge=pd.merge(df1,df2,on='first_name',how=inner)=default inner .left,right,outer
'''
sample.head(3)
sample.columns
# distribution of number of booking attempts
sample.groupby('user_id')['is_booking']\
   .agg({'num_of_bookings':'count'}).reset_index()\
   .groupby('num_of_bookings')['user_id']\
   .agg('count')
#merging a column count
sample = sample.merge(sample.groupby('user_id')['is_booking']
    .agg(['count']).reset_index())

# distribution of booking rate
sample.groupby('user_id')['is_booking']\
   .agg(['mean']).reset_index()\
   .groupby('mean')['user_id']\
   .agg('count')
sample['is_booking'].sum()
#####################explore the data - validate data##################################

#number of guests need to be > 0
pd.crosstab(sample['srch_adults_cnt'], sample['srch_children_cnt'])  #174 is having zero values
sample.drop(sample[sample['srch_adults_cnt'] + sample['srch_children_cnt']==0].index) # index metod finds the index

type(sample['srch_co'])
sample['srch_co'] = pd.to_datetime(sample['srch_co']) #to_datetime converts argument in to date and time
sample['srch_ci'] = pd.to_datetime(sample['srch_ci'])
sample['date_time'] = pd.to_datetime(sample['date_time'])
sample['date'] = pd.to_datetime(sample['date_time'].apply(lambda x: x.date()))
sample.head()
# Check-out date need to be later than check-in date;
# Check-in date need to be later than booking date

sample[sample['srch_co'] < sample['srch_ci']][['srch_co', 'srch_ci']]
sample[sample['srch_ci'] < sample['date']][['srch_ci', 'date']]
sample.shape
sample.head()

#####################explore the data - create new variables that might be useufl###########
def duration(row):
    delta = (row['srch_co'] - row['srch_ci'])/np.timedelta64(1, 'D') #divide by one day
    if delta <= 0:
        return np.nan
    else:

        return delta
def days_in_advance(row):
    delta = (row['srch_ci'] - row['date'])/np.timedelta64(1, 'D') 
    if delta < 0:
        return np.nan
    else:
        return delta


sample['duration'] = sample.apply(duration, axis=1)
sample['days_in_advance'] = sample.apply(days_in_advance, axis=1)
'''POPULATION PROPORTION
When we have a categorical variable of interest measured in two populations,
 it is quite often that we are interested in comparing the proportions of
 a certain category for the the two populations.

hypothesis test for the  equality of the booking rate in two binomial samples
(one segment and other segment)is booking rate for city1 greater than other cities?
ho  p1=p2
ha  p1!=p2
z=p1-p2/sqrt(p(1-p)(1/n1+1/n2))

'''
pd.set_option("display.max_columns",10)
############## Outperforming/underperforming segments - two sample t test #############
def stats_comparison(i):
    sample.groupby(i)['is_booking'].agg({
    'average': 'mean',
    'bookings': 'count'
    }).reset_index()
    cat = sample.groupby(i)['is_booking']\
        .agg({
            'sub_average': 'mean',
            'sub_bookings': 'count'
       }).reset_index()
    cat['overall_average'] = sample['is_booking'].mean()
    cat['overall_bookings'] = sample['is_booking'].count()
    cat['rest_bookings'] = cat['overall_bookings'] - cat['sub_bookings']
    cat['rest_average'] = (cat['overall_bookings']*cat['overall_average'] \
                     - cat['sub_bookings']*cat['sub_average'])/cat['rest_bookings']
    cat['z_score'] = (cat['sub_average']-cat['rest_average'])/\
        np.sqrt(cat['overall_average']*(1-cat['overall_average'])
            *(1/cat['sub_bookings']+1/cat['rest_bookings']))
    cat['prob'] = np.around(stats.norm.cdf(cat.z_score), decimals = 10)
    cat['significant'] = [(lambda x: 1 if x > 0.1 else -1 )(i) for i in cat['prob']]
    print(cat.head)
   
stats_comparison('user_location_city')
stats_comparison('channel')

############## clustering - what are the similar user cities? ##############

# Step 1: what are the features I am going to use (that make sense)?
# What features may distinguish cities? based on business sense and exploratory analysis

num_list = ['duration', 'days_in_advance', 'orig_destination_distance', 'is_mobile',
            'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt']
city_data = sample.dropna(axis=0)[num_list + ['user_location_city']]             #removes na value in rows
city_groups = city_data.groupby('user_location_city').mean().reset_index().dropna(axis=0)

city_data.head()
city_groups.head()



# Step 2: shall I standardise the data?
# What is the magnitude of data range?

city_groups_std = city_groups.copy()
for i in num_list:
    city_groups_std[i] = preprocessing.scale(city_groups_std[i])


city_groups_std.head()

#using the elbow method to find the optimal number of cluster
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++') #kmeans ++ used to correctly select the cluster
    kmeans.fit(city_data[num_list])
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('number of cluster')
plt.ylabel('wcss')
plt.show()

#applying kmeans
km=KMeans(n_clusters=3,init='k-means++')
city_groups_std['cluster']=km.fit_predict(city_groups_std[num_list])




#appling pca to visualize
pca=decomposition.PCA(n_components=2)
pca.fit(city_groups[num_list])
city_groups_std['x']=pca.fit_transform(city_groups_std[num_list])[:,0]
city_groups_std['y']=pca.fit_transform(city_groups_std[num_list])[:,1]
plt.scatter(city_groups_std['x'],city_groups_std['y'],c=city_groups_std['cluster'])
plt.show()

city_groups_std.head()
# merging the two dataframes based on a common column user_location_city
city_groups.merge(city_groups_std[['user_location_city', 'cluster']])\
    .groupby('cluster')\
    .mean() 

#################decision tree - what lead to higher chance of booking for indviduals?##########################
from sklearn.cross_validation import train_test_split

#choose a cluster and split them into test and train
sample = sample.merge(city_groups_std[['user_location_city', 'cluster']], left_on='user_location_city', right_on='user_location_city', how='outer')
sample.groupby('cluster')['is_booking'].count()
sample.head()

# choose one of the city clusters to analyze
from sklearn.cross_validation import train_test_split
tree_data = sample.dropna(axis = 0)
tree_train, tree_test = train_test_split(tree_data, test_size=0.2, random_state=1, stratify=tree_data['is_booking'])

#build the decision tree model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_leaf_nodes=6,min_samples_leaf=200)
clf=clf.fit(tree_train[num_list],tree_train['is_booking'])

# scoring of the prediction model
clf.score(tree_test[num_list], tree_test['is_booking'])

# visualize the decision tree
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names =['duration', 'days_in_advance', 'orig_destination_distance', 'is_mobile',
            'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt'], filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("booking3_tree.pdf")







