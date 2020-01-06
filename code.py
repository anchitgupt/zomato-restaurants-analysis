#!/usr/bin/env python
# coding: utf-8

# ## Name  : Anchit Gupta
# ## Roll no: MT19060


# In[1]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[2]:


import sys
print(sys.executable)


# In[3]:


# Libraries used
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import pandas_profiling as pp
from IPython.core.display import display, HTML
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,f1_score, recall_score,precision_score
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import (LinearRegression, Ridge, Lasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
get_ipython().run_line_magic('matplotlib', 'inline')
tqdm().pandas()


# In[4]:


src_file = "zomato.csv"
data = pd.read_csv(src_file)


# In[5]:


data.info()


# In[6]:


display(HTML('<h1>Columns data type</h1>'))
data_columns = data.columns.tolist()
for i in data_columns:
    print(i," : ",  data[i].dtype)


# In[7]:


# Renaming the column values
column_name = {'usl':'url', 
               'address':'address',
               'name':'name',  
               'online_order':'netorder',
               'book_table':'booktable', 
               'rate':'rating',
               'votes':'votes', 
               'phone':'phone', 
               'location':'location', 
               'rest_type':'resttype', 
               'dish_liked':'dishliked', 
               'cuisines':'cuisines', 
               'approx_cost(for two people)':'cost', 
               'reviews_list':'reviews', 
               'menu_item':'menuitems', 
               'listed_in(type)':'listedin', 
               'listed_in(city)':'city'}


# In[8]:


data  = data.rename(columns= column_name)


# In[9]:


data.isnull().sum()


# In[10]:


display(HTML('<h1>Percentage of Null Values</h1>'))
data.isnull().sum()*100/data.shape[0]


# In[11]:

data.head()


# In[12]:

data.describe()



# ### Here Items are pruned on basis that usability and necessity to contribute in analysis

# In[13]:


prune_items = ['url', 'address', 'name', 'phone', 'location',  'dishliked', 'menuitems']
data =  data.drop(prune_items, axis=1)


# In[14]:


# Removing the rating
data.rating = data.rating.str[:3]


# In[15]:


# Storing the data temporary
df = data[:]


# ### Function to fill empty rating values from the reviews by rating

# In[16]:


def rating_is_null():
    reviews= {}
    null_brackets = []
    rating_null_index = df[df['rating'].isnull()].index.tolist()
    #print('null index: ', rating_null_index)
    for null_indexes in rating_null_index:
        temp_rev = df['reviews'][null_indexes]
        if temp_rev == '[]':
            null_brackets.append(null_indexes)
        elif type(temp_rev) == str:
            review_index = [i for i in range(len(temp_rev)) if temp_rev.startswith('(\'Rated ', i)]
            scores = [float(temp_rev[i+7:i+10]) for i in review_index]
            #print(np.mean(scores))
            df.loc[null_indexes, 'rating'] = round(np.mean(scores), 2)
    return null_brackets
    
    


# In[17]:


rating_still_null = rating_is_null()


# In[18]:


df['rating'] =  df['rating'].replace( to_replace =['NEW', '-'], value =np.nan)
df['rating'] =  df['rating'].fillna(df['rating'].mode()[0])
df['rating'] =  df['rating'].astype(float)
df['rating'] =  pd.cut(df['rating'], 4, labels=[1,2,3,4])
df['rating'].unique()


# In[19]:


df = df.dropna()

# TODO

df = df.drop('reviews', axis=1)
df = df.reset_index()


# In[20]:


df.isnull().sum()


# ### Creating asymmetric data of compound columns

# In[21]:


def createSimpleColumns(c_name):
    restype_unique = df[c_name].unique().tolist()
    final_resttype = []
    for i in restype_unique:
        if(i.find(',') != -1):
            temp = i.split(',')
            for j in temp:
                final_resttype.append(j.strip().replace(" ", "_"))
        else:
            final_resttype.append(i.strip().replace(" ", "_"))
            
    final_resttype = list(frozenset(final_resttype))
    restype_datalist = df[c_name].tolist()
    df_resttype = pd.DataFrame(0, index=np.arange(len(restype_datalist)),columns=final_resttype)
    c = 0

    for i in tqdm(restype_datalist):
        k = []
        if (i.find(',') != -1):
            temp = i.split(',')
            for j in temp:
                j = j.strip()
                k.append(j)
        else:
            k.append(i.strip())
        df_resttype.loc[c] = np.isin(final_resttype, k).astype(int)
        c+=1
    return df_resttype, final_resttype


# In[22]:


df_resttype,list_resttype = createSimpleColumns('resttype')
df_cuisines, list_cuisines = createSimpleColumns('cuisines')
df_resttype.rename(columns={'Cafe':'Rest_Cafe', 'Bakery':'Rest_Bakery'}, inplace=True)
d_res_cus = pd.merge(df_resttype, df_cuisines, left_index=True, right_index=True)
df = pd.merge(df, d_res_cus, left_index=True, right_index=True)


# In[23]:


df  = df.drop('resttype', axis = 1)
df  = df.drop('cuisines', axis = 1)
# df  = df.drop(['level_0','index'], axis = 1)
df = df.drop('index', axis=1)


# In[24]:


df['netorder']  = df['netorder'].replace(to_replace=['No', 'Yes'], value=[0, 1])
df['booktable'] = df['booktable'].replace(to_replace=['No', 'Yes'], value=[0, 1])


# In[25]:


df.head()


# In[26]:


toonedata = df['listedin'].unique().tolist()


# In[27]:


df_listedin, list_listedin = createSimpleColumns('listedin')
df_city, list_city     = createSimpleColumns('city')
df          = pd.merge(df, df_listedin, left_index=True, right_index=True)
df          = pd.merge(df, df_city    , left_index=True, right_index=True)
df          = df.drop('city', axis=1)
df          = df.drop('listedin', axis=1)


# ### Rearranging the columns

# In[28]:


final_columns =  df.columns.tolist()[3:] +df.columns.tolist()[:3]
df = df[final_columns]


# In[29]:


df.head()


# In[30]:


#plt.figure(figsize=(10,10))
outlets = df['rating'].value_counts()
sns.barplot(x = outlets.index, y = outlets)
plt.title("Restaurdents getting ratings")
plt.ylabel('Frequency of Restaurants getting ratings')
plt.xlabel('Ratings')
plt.savefig("rating-classes-frequency.png")
plt.show()


# In[31]:



# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.1, 0.1, 0.1, 0.1)
#add colors
fig1, ax1 = plt.subplots()
ax1.pie(outlets, explode=explode, labels=outlets.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.savefig("pie-rating-classes-percentage.png")
plt.show()


# In[32]:


def topcuisine():
    k =[i for i in list_cuisines if i in df.columns]
    list_cuisines_top = {}
    for i in k:
        if df[i].value_counts()[0] != df.shape[0]:
            list_cuisines_top[i]=df[i].value_counts().tolist()[1]
    list_cuisines_top_sorted = sorted(list_cuisines_top.items(), key=lambda x:x[-1],reverse=True)
    plt.figure(figsize=(20,10))
    sns.barplot(x = [i[0] for i in list_cuisines_top_sorted[:20]], y = [i[1] for i in list_cuisines_top_sorted[:20]])
    plt.title("Top 20 Cuisines")
    plt.ylabel('Frequency of Restaurants having these cuisines')
    plt.xlabel('Cuisine')
    plt.savefig("top20-cuisines.png")
    plt.show()

topcuisine()     


# In[33]:


def topresttype():
    k =[i for i in list_resttype if i in df.columns]
    list_resttype_top = {}
    for i in k:
        if df[i].value_counts()[0] != df.shape[0]:
            list_resttype_top[i]=df[i].value_counts().tolist()[1]
    
    list_resttype_top_sorted = sorted(list_resttype_top.items(), key=lambda x:x[-1],reverse=True)
    list_top_rest = [i[0] for i in list_resttype_top_sorted[:20]]
    plt.figure(figsize=(20,10))
    sns.barplot(x = [i[0] for i in list_resttype_top_sorted[:20]], y = [i[1] for i in list_resttype_top_sorted[:20]])
    plt.title("Top 20 Restaurants Types")
    plt.ylabel('Frequency of types Restaurants')
    plt.xlabel('Types of Restaurants')
    plt.savefig("top20-restaurdants-type.png")
    plt.show()
    return list_top_rest

list_top_rest = topresttype() 


# In[34]:


def topresttopcity():
    
    k =[i for i in list_city if i in df.columns]
    list_city_top = {}
    for i in k:
        if df[i].value_counts()[0] != df.shape[0]:
            list_city_top[i]=df[i].value_counts().tolist()[1]
    list_city_top_sorted = sorted(list_city_top.items(), key=lambda x:x[-1],reverse=True)
    top_city_name = [i[0] for i in list_city_top_sorted[:10]]
    topresttopcity_data = []
    
    matrix_city_rest = pd.DataFrame(index=top_city_name, columns=list_top_rest)
    for i in top_city_name:
        for j in list_top_rest:
            l = [i,j]
            d = df[l]
            d = d.sum(axis=1).to_frame()
            sup = (d[0] == 2).sum()
            matrix_city_rest[j].loc[i] = sup
            topresttopcity_data.append([i,j,sup])
    plt.figure(figsize=(10,10))
    plt.title("Top 10 Crowded cities having the top restaurants types")
    sns.heatmap(matrix_city_rest, annot=True, fmt="g", cmap='viridis')
    plt.savefig("top10-topcity-toprestaurants.png")
    plt.show()
    
topresttopcity()


# In[35]:


df['rating'].value_counts()


# ### Encoding the data

# In[36]:


labelencoder   = LabelEncoder()


# In[37]:


df = df.apply(labelencoder.fit_transform)


# In[38]:


df2 = df[:]


# In[39]:


#Using Pearson Correlation
corr = df.corr().abs()
corr


# In[40]:


to_drop = []
for col in df.columns:
    if np.isnan(corr[col].values).astype(int).sum() == df.shape[1] :
        to_drop.append(col)


# In[41]:


df = df.drop(to_drop, axis=1)


# In[42]:


def drawranking(r, n, order=1):
    return dict(zip(n, map(lambda x: round(x,2),  MinMaxScaler().fit_transform(order*np.array([r]).T).T[0])))


# In[43]:


b_df = df[:]
cols = list(df)
feature_cols = cols[:len(cols)-1]
X = df[feature_cols] # Features
y = df.rating # Target variable


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[45]:


def getModels(X,y):
    colnames = feature_cols
    print(len(colnames))
    model = [0,0,0,0,0]
    name  = ['LinearRegression', 'RFE', 'Ridge', 'Lasso', 'RandomForestRegressor']
    model[0]  = LinearRegression(normalize=True).fit(X, y)
    model[1]  = RFE(model[0], verbose =3).fit(X, y)
    model[2]  = Ridge(alpha = 7).fit(X, y)
    model[3]  = Lasso(alpha=0.05).fit(X, y)
    model[4]  = RandomForestRegressor(n_jobs=-1, n_estimators=100, verbose=3).fit(X, y)
    return model
model = getModels(X,y)


# In[ ]:


def drawgraph(data,gph_columns):
    sns.catplot(x=gph_columns[1], y=gph_columns[0], data = data[:20], kind="bar",height=10,  palette='coolwarm')
    plt.title("Top 20")
    plt.savefig("top20-features.png")
    sns.catplot(x=gph_columns[1], y=gph_columns[0], data = data[-20:], kind="bar",height=10,  palette='coolwarm')
    plt.title("Bottom 20")
    plt.savefig("bottom20-features.png")
    


# In[46]:



    
def testattributes(X,y):
    ranks = {}
    name  = ['LinearRegression', 'RFE', 'Ridge', 'Lasso', 'RandomForestRegressor']
    for i in range(len(model)):
        if name[i] == 'RFE':
            ranks[name[i]] =  drawranking(list(map(float, model[i].ranking_)), feature_cols, order=-1)
        elif name[i] == 'RandomForestRegressor':
            ranks[name[i]] =  drawranking(model[i].feature_importances_, feature_cols)
        else:
            ranks[name[i]] =  drawranking(list(map(float, model[i].coef_)), feature_cols)
   
    mean_scores = {}
    for i in feature_cols:
        mean_scores[i] = round(np.mean([ranks[method][i] for method in ranks.keys()]), 4)

    gph_columns = ['Feature','Mean Ranking']
    data = pd.DataFrame(list(mean_scores.items()), columns=gph_columns).sort_values(gph_columns[1], ascending=False)
    drawgraph(data,gph_columns)
    return data
meanplot = testattributes(X,y)


# In[47]:


from xgboost import plot_importance

def xgaimportancefilter(X,y):
    model = XGBClassifier()
    model.fit(X, y)
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_importance(model, ax=ax)
    plt.savefig("feature-importance-with-feature-removing.png")
    plt.show()


# In[48]:


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# In[49]:


modelNB = GaussianNB()
rfc = RandomForestClassifier(n_estimators=50)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
xgb = XGBClassifier()

models = [modelNB, rfc, mlp, xgb]
modelNames = ['GaussianNB','RandomForest', 'MLPC', 'XGBClassifier']


# In[50]:


feature_are = ['Dropping Least Attribute', 'Original Attribute', 'Balanced Class']
modelWithDiff = [[],[],[]]
def getModelScores(type):
    l = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(modelNames[models.index(model)]," Accuracy:",accuracy_score(y_test, y_pred)) 
        l.append([type,modelNames[models.index(model)],accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')])
    return l


# In[51]:


to_drop = meanplot[meanplot['Mean Ranking'] <= 0.35]['Feature'].tolist()
df2 = df[:]
df2 = df.drop(to_drop, axis=1)


# In[52]:


cols = list(df2)
feature_cols = cols[:len(cols)-1]
X = df2[feature_cols] # Features
y = df2.rating # Target variable


# In[53]:


modelWithDiff[0] = getModelScores(feature_are[0])


# In[54]:


# xgaimportancefilter(X,y)


# In[55]:


cols = list(df)
feature_cols = cols[:len(cols)-1]
X = df[feature_cols] # Features
y = df.rating # Target variable


# In[56]:


modelWithDiff[1] = getModelScores(feature_are[1])


# In[57]:


model = XGBClassifier()
model.fit(X, y)


# In[58]:


fig, ax = plt.subplots(figsize=(10, 10))
plot_importance(model, ax=ax)
plt.savefig("feature-importance.png")
plt.show()


# In[59]:


df_temp = df[:]
d_rate2 = resample(df[df['rating'] == 2], replace=False, n_samples=1200, random_state=123)
d_rate3 = resample(df[df['rating'] == 3], replace=False, n_samples=1200, random_state=123)
d_rate1 = resample(df[df['rating'] == 1], replace=False, n_samples=1200, random_state=123)
d_rate0 = resample(df[df['rating'] == 0], replace=False, n_samples=640, random_state=123)
df_sampled = pd.concat([d_rate0,d_rate1,d_rate2,d_rate3])


# In[60]:


cols = list(df_sampled)
feature_cols = cols[:len(cols)-1]
X = df_sampled[feature_cols] # Features
y = df_sampled.rating # Target variable


# In[61]:


modelWithDiff[2] = getModelScores(feature_are[2])


# In[62]:


xgaimportancefilter(X,y)


# In[63]:


data_m_plot = []
for i in modelWithDiff:
    for j in i:
        data_m_plot.append(j)


# In[64]:


dfmodelcomp = pd.DataFrame(data = data_m_plot, columns=['Attribute_Used', 'Models', 'Accuracy', 'Precision Score','F1-Score'])
dfmodelcomp


# In[65]:


sns.factorplot('Attribute_Used','Accuracy', col='Models',data=dfmodelcomp, kind='bar')
plt.savefig("modelsAccuracy.png")


# In[66]:


sns.factorplot('Attribute_Used','Precision Score', col='Models',data=dfmodelcomp, kind='bar')
plt.savefig("modelsPrecision.png")


# In[67]:


sns.factorplot('Attribute_Used','F1-Score', col='Models',data=dfmodelcomp, kind='bar')
plt.savefig("modelsf1score.png")


# In[ ]:




