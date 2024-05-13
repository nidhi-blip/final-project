from altair import DataFrameLike
import streamlit as st
import itertools
import pandas as pd 
import numpy as np 
from scipy.stats import pearsonr
from scipy import stats
import matplotlib.pyplot as plt 
import plotly.express as px
import matplotlib
import seaborn as sns
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from scipy.stats import chi2_contingency,chi2
import statsmodels.api as sm 
from scipy.stats import spearmanr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from scipy.stats import anderson
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from PIL import Image

#############
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)

import base64
import io
###############
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

image = Image.open('cover.jpg')
matplotlib.use("Agg")

class DataFrame_Loader():

    def __init__(self):
        print("Loadind DataFrame")

    def read_csv(self, data):
        self.df = pd.read_csv(data)
        return self.df


class EDA_Dataframe_Analysis():

    def __init__(self):
        print("General_EDA object created")

    def show_dtypes(self, x):
        return x.dtypes

    def show_columns(self, x):
        return x.columns

    def Show_Missing(self, x):
        return x.isna().sum()

    def Show_Missing1(self, x):
        return x.isna().sum()

    def Show_Missing2(self, x):
        return x.isna().sum()

    def show_hist(self, x):
        return x.hist()

    def Tabulation(self, x):
        table = pd.DataFrame(x.dtypes, columns=['dtypes'])
        table1 = pd.DataFrame(x.columns, columns=['Names'])
        table = table.reset_index()
        table = table.rename(columns={'index': 'Name'})
        table['No of Missing'] = x.isnull().sum().values
        table['No of Uniques'] = x.nunique().values
        table['Percent of Missing'] = ((x.isnull().sum().values) / (x.shape[0])) * 100
        table['First Observation'] = x.loc[0].values
        table['Second Observation'] = x.loc[1].values
        table['Third Observation'] = x.loc[2].values
        for name in table['Name'].value_counts().index:
            table.loc[table['Name'] == name, 'Entropy'] = round(stats.entropy(x[name].value_counts(normalize=True), base=2), 2)
        return table

    def Numerical_variables(self, x):
        Num_var = [var for var in x.columns if x[var].dtypes != "object"]
        Num_var_df = x[Num_var]
        return Num_var_df

    def categorical_variables(self, x):
        cat_var = [var for var in x.columns if x[var].dtypes == "object"]
        cat_var = x[cat_var]
        return cat_var

    def impute(self, x):
        df = x.dropna()
        return df

    def imputee(self, x):
        df = x.dropna()
        return df

    def Show_pearsonr(self, x, y):
        result = pearsonr(x, y)
        return result

    

    def plotly(self, a, x, y):
        fig = px.scatter(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
        fig.show()

    def show_displot(self, x):
        plt.figure(1)
        plt.subplot(121)
        sns.distplot(x)

        plt.subplot(122)
        x.plot.box(figsize=(16, 5))

        plt.show()

    def Show_DisPlot(self, x):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(12, 7))
        return sns.distplot(x, bins=25)

    def Show_CountPlot(self, x):
        fig_dims = (18, 8)
        fig, ax = plt.subplots(figsize=fig_dims)
        return sns.countplot(x, ax=ax)

    def plotly_histogram(self, a, x, y):
        fig = px.histogram(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
        fig.show()

    def plotly_violin(self, a, x, y):
        fig = px.histogram(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
        fig.show()

    def Show_PairPlot(self, x):
        return sns.pairplot(x)

    def Show_HeatMap(self, x):
        f, ax = plt.subplots(figsize=(15, 15))
        return sns.heatmap(x.corr(), annot=True, ax=ax)

    def wordcloud(self, x):
        wordcloud = WordCloud(width=1000, height=500).generate(" ".join(x))
        plt.imshow(wordcloud)
        plt.axis("off")
        return wordcloud

    def label(self, x):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        x = le.fit_transform(x)
        return x

    def label1(self, x):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        x = le.fit_transform(x)
        return x

    def concat(self, x, y, z, axis):
        return pd.concat([x, y, z], axis)

    def dummy(self, x):
        return pd.get_dummies(x)

    def qqplot(self, x):
        return sm.qqplot(x, line='45')

    def Anderson_test(self, a):
        return anderson(a)

    def PCA(self, x):
        pca = PCA(n_components=8)
        principlecomponents = pca.fit_transform(x)
        principledf = pd.DataFrame(data=principlecomponents)
        return principledf

    def outlier(self, x):
        high = 0
        q1 = x.quantile(.25)
        q3 = x.quantile(.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high += q3 + 1.5 * iqr
        outlier = (x.loc[(x < low) | (x > high)])
        return outlier

    def check_cat_relation(self, x, y, confidence_interval):
        cross_table = pd.crosstab(x, y, margins=True)
        stat, p, dof, expected = chi2_contingency(cross_table)
        print("Chi_Square Value = {0}".format(stat))
        print("P-Value = {0}".format(p))
        alpha = 1 - confidence_interval
        return p, alpha
        if p > alpha:
            print(">> Accepting Null Hypothesis <<")
            print("There Is No Relationship Between Two Variables")
        else:
            print(">> Rejecting Null Hypothesis <<")
            print("There Is A Significance Relationship Between Two Variables")




class Attribute_Information():

    def __init__(self):
        
        print("Attribute Information object created")
        
    def Column_information(self,data):
    
        data_info = pd.DataFrame(
                                columns=['No of observation',
                                        'No of Variables',
                                        'No of Numerical Variables',
                                        'No of Factor Variables',
                                        'No of Categorical Variables',
                                        'No of Logical Variables',
                                        'No of Date Variables',
                                        'No of zero variance variables'])


        data_info.loc[0,'No of observation'] = data.shape[0]
        data_info.loc[0,'No of Variables'] = data.shape[1]
        data_info.loc[0,'No of Numerical Variables'] = data._get_numeric_data().shape[1]
        data_info.loc[0,'No of Factor Variables'] = data.select_dtypes(include='category').shape[1]
        data_info.loc[0,'No of Logical Variables'] = data.select_dtypes(include='bool').shape[1]
        data_info.loc[0,'No of Categorical Variables'] = data.select_dtypes(include='object').shape[1]
        data_info.loc[0,'No of Date Variables'] = data.select_dtypes(include='datetime64').shape[1]
        data_info.loc[0,'No of zero variance variables'] = data.loc[:,data.apply(pd.Series.nunique)==1].shape[1]

        data_info =data_info.transpose()
        data_info.columns=['value']
        data_info['value'] = data_info['value'].astype(int)


        return data_info

    def __get_missing_values(self,data):
        
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        #Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)
        
        #Returning missing values
        return missing_values

        
    def __iqr(self,x):
        return x.quantile(q=0.75) - x.quantile(q=0.25)

    def __outlier_count(self,x):
        upper_out = x.quantile(q=0.75) + 1.5 * self.__iqr(x)
        lower_out = x.quantile(q=0.25) - 1.5 * self.__iqr(x)
        return len(x[x > upper_out]) + len(x[x < lower_out])

    def num_count_summary(self,df):
        df_num = df._get_numeric_data()
        data_info_num = pd.DataFrame()
        i=0
        for c in  df_num.columns:
            data_info_num.loc[c,'Negative values count']= df_num[df_num[c]<0].shape[0]
            data_info_num.loc[c,'Positive values count']= df_num[df_num[c]>0].shape[0]
            data_info_num.loc[c,'Zero count']= df_num[df_num[c]==0].shape[0]
            data_info_num.loc[c,'Unique count']= len(df_num[c].unique())
            data_info_num.loc[c,'Negative Infinity count']= df_num[df_num[c]== -np.inf].shape[0]
            data_info_num.loc[c,'Positive Infinity count']= df_num[df_num[c]== np.inf].shape[0]
            data_info_num.loc[c,'Missing Percentage']= df_num[df_num[c].isnull()].shape[0]/ df_num.shape[0]
            data_info_num.loc[c,'Count of outliers']= self.__outlier_count(df_num[c])
            i = i+1
        return data_info_num
    
    def statistical_summary(self,df):
    
        df_num = df._get_numeric_data()

        data_stat_num = pd.DataFrame()

        try:
            data_stat_num = pd.concat([df_num.describe().transpose(),
                                       pd.DataFrame(df_num.quantile(q=0.10)),
                                       pd.DataFrame(df_num.quantile(q=0.90)),
                                       pd.DataFrame(df_num.quantile(q=0.95))],axis=1)
            data_stat_num.columns = ['count','mean','std','min','25%','50%','75%','max','10%','90%','95%']
        except:
            pass

        return data_stat_num
class Classification_Analyses():
	def __init__(self):
		print("Classification object created")

	def filedownload(self,df, filename):
		csv = df.to_csv(index=False)
		b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
		href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
		return href

	def imagedownload(self,plt, filename):
		s = io.BytesIO()
		plt.savefig(s, format='pdf', bbox_inches='tight')
		plt.close()
		b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
		href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
		return href
	def build_model(self,df,seed_number,split_size):
		df = df.loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
		X = df.iloc[:,:-1] # Using all column except for the last column as X
		Y = df.iloc[:,-1] # Selecting the last column as Y

		st.markdown('**1.2. Dataset dimension**')
		st.write('X')
		st.info(X.shape)
		st.write('Y')
		st.info(Y.shape)

		st.markdown('**1.3. Variable details**:')
		st.write('X variable (first 20 are shown)')
		st.info(list(X.columns[:20]))
		st.write('Y variable')
		st.info(Y.name)

		# Build lazy model
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
		#reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
		clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None,predictions=True)
		models_train,predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
		models_test,predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)

		# st.subheader('2. Table of Model Performance')

		# st.write('Training set')
		# st.write(predictions_train)
		# st.markdown(self.filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

		# st.write('Test set')
		# st.write(predictions_test)
		# st.markdown(self.filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

		st.subheader('3. Plot of Model Performance')
		st.write(models_train)
		st.subheader('Plot of Model Performance(Test Set)')
		st.write(models_test)



		with st.markdown('**Accuracy Chart**'):
			# Tall
			# predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"] ]
			
		# st.markdown(self.imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
			# Wide
		# plt.figure(figsize=(9, 3))
		# sns.set_theme(style="whitegrid")
		# ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
		# ax1.set(ylim=(0, 1))
		# plt.xticks(rotation=90)
		# st.pyplot(plt)
			plt.figure(figsize=(5, 10))
			sns.set_theme(style="whitegrid")
			ax = sns.barplot(y=models_train.index, x="Accuracy", data=models_train)
			st.pyplot(plt)
		st.markdown(self.imagedownload(plt,'Accuracy.pdf'), unsafe_allow_html=True)


		with st.markdown('**Accuracy Chart!!**'):
			# plt.figure(figsize=(10, 5))
			# sns.set_theme(style="whitegrid")
			# ax = sns.barplot(x=models_train.index,y="Accuracy", data=models_train)
			
			plt.figure(figsize=(10, 5))
			sns.set_theme(style="whitegrid")
			ax = sns.barplot(x=models_train.index, y="Accuracy", data=models_train)
			plt.xticks(rotation=90)
			st.pyplot(plt)

		st.markdown(self.imagedownload(plt,'Accuracy.pdf'), unsafe_allow_html=True)

		# with st.markdown('**RMSE (capped at 50)**'):
		# 	# Tall
		# 	predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"] ]
		# 	plt.figure(figsize=(3, 9))
		# 	sns.set_theme(style="whitegrid")
		# 	ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
		# # st.markdown(self.imagedownload(plt,'plot-rmse-tall.pdf'), unsafe_allow_html=True)
		# 	# Wide
		# plt.figure(figsize=(9, 3))
		# sns.set_theme(style="whitegrid")
		# ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
		# plt.xticks(rotation=90)
		# st.pyplot(plt)
		# # st.markdown(self.imagedownload(plt,'plot-rmse-wide.pdf'), unsafe_allow_html=True)

		# with st.markdown('**Calculation time**'):
		# 	# Tall
		# 	# predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
		# 	plt.figure(figsize=(3, 9))
		# 	sns.set_theme(style="whitegrid")
		# 	ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
		# # st.markdown(self.imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
		# 	# Wide
		# plt.figure(figsize=(9, 3))
		# sns.set_theme(style="whitegrid")
		# ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
		# plt.xticks(rotation=90)
		# st.pyplot(plt)
		# st.markdown(self.imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)




class ML_Analyses():
	def __init__(self):
		print("ML Automation object created")
	
	# Download CSV data
	
	def filedownload(self,df, filename):
		csv = df.to_csv(index=False)
		b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
		href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
		return href

	def imagedownload(self,plt, filename):
		s = io.BytesIO()
		plt.savefig(s, format='pdf', bbox_inches='tight')
		plt.close()
		b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
		href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
		return href

		# Model building
	def build_model(self,df,split_size,seed_number):
		df = df.loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
		X = df.iloc[:,:-1] # Using all column except for the last column as X
		Y = df.iloc[:,-1] # Selecting the last column as Y

		st.markdown('**1.2. Dataset dimension**')
		st.write('X')
		st.info(X.shape)
		st.write('Y')
		st.info(Y.shape)

		st.markdown('**1.3. Variable details**:')
		st.write('X variable (first 20 are shown)')
		st.info(list(X.columns[:20]))
		st.write('Y variable')
		st.info(Y.name)

		# Build lazy model
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
		reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
		models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
		models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

		st.subheader('2. Table of Model Performance')

		st.write('Training set')
		# st.write(predictions_train)
		# st.markdown(self.filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)
		st.write(models_train)
		st.markdown(self.filedownload(models_train,'training.csv'), unsafe_allow_html=True)

		st.write('Test set')
		# st.write(predictions_test)
		# st.markdown(self.filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)
		st.write(models_test)
		st.markdown(self.filedownload(models_test,'test.csv'), unsafe_allow_html=True)

		st.subheader('3. Plot of Model Performance (Test set)')


		with st.markdown('**R-squared**'):
			# Tall
			predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"] ]
			plt.figure(figsize=(3, 9))
			sns.set_theme(style="whitegrid")
			ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
			ax1.set(xlim=(0, 1))
		st.markdown(self.imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
			# Wide
		plt.figure(figsize=(9, 3))
		sns.set_theme(style="whitegrid")
		ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
		ax1.set(ylim=(0, 1))
		plt.xticks(rotation=90)
		st.pyplot(plt)
		st.markdown(self.imagedownload(plt,'plot-r2-wide.pdf'), unsafe_allow_html=True)

		with st.markdown('**RMSE (capped at 50)**'):
			# Tall
			predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"] ]
			plt.figure(figsize=(3, 9))
			sns.set_theme(style="whitegrid")
			ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
		st.markdown(self.imagedownload(plt,'plot-rmse-tall.pdf'), unsafe_allow_html=True)
			# Wide
		plt.figure(figsize=(9, 3))
		sns.set_theme(style="whitegrid")
		ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
		plt.xticks(rotation=90)
		st.pyplot(plt)
		st.markdown(self.imagedownload(plt,'plot-rmse-wide.pdf'), unsafe_allow_html=True)

		with st.markdown('**Calculation time**'):
			# Tall
			predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
			plt.figure(figsize=(3, 9))
			sns.set_theme(style="whitegrid")
			ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
		st.markdown(self.imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
			# Wide
		plt.figure(figsize=(9, 3))
		sns.set_theme(style="whitegrid")
		ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
		plt.xticks(rotation=90)
		st.pyplot(plt)
		st.markdown(self.imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)


class Data_Base_Modelling():

    
    def __init__(self):
        
        print("General_EDA object created")


class YourClass:
    def Label_Encoding(self, x):
        category_col = [var for var in x.columns if x[var].dtypes == "object"]
        labelEncoder = preprocessing.LabelEncoder()
        mapping_dict = {}
        for col in category_col:
            x[col] = labelEncoder.fit_transform(x[col])
            le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
            mapping_dict[col] = le_name_mapping
        return mapping_dict

    def IMpupter(self, x):
        imp_mean = IterativeImputer(random_state=0)
        x = imp_mean.fit_transform(x)
        x = pd.DataFrame(x)
        return x

    def Logistic_Regression(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', LogisticRegression())])
        pipelines = [pipeline_dt]
        for pipe in pipelines:
            pipe.fit(x_train, y_train)
        for i, model in enumerate(pipelines):
            return classification_report(y_test, model.predict(x_test))

    def Decision_Tree(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', DecisionTreeClassifier())])
        pipelines = [pipeline_dt]
        for pipe in pipelines:
            pipe.fit(x_train, y_train)
        for i, model in enumerate(pipelines):
            return classification_report(y_test, model.predict(x_test))

    def RandomForest(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', RandomForestClassifier())])
        pipelines = [pipeline_dt]
        for pipe in pipelines:
            pipe.fit(x_train, y_train)
        for i, model in enumerate(pipelines):
            return classification_report(y_test, model.predict(x_test))

    def naive_bayes(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', GaussianNB())])
        pipelines = [pipeline_dt]
        for pipe in pipelines:
            pipe.fit(x_train, y_train)
        for i, model in enumerate(pipelines):
            return classification_report(y_test, model.predict(x_test))

    def XGb_classifier(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', XGBClassifier())])
        pipelines = [pipeline_dt]
        for pipe in pipelines:
            pipe.fit(x_train, y_train)
        for i, model in enumerate(pipelines):
            return classification_report(y_test, model.predict(x_test))


    

st.image(image, use_column_width=True)  
def main():
    st.title("Machine Learning Application for Automated EDA")
    
    st.info("This Web Application is created to automate EDA and ML") 
    activities = ["General EDA", "EDA For Linear Models", "ML Analysis Report", "Classification Analysis Report"]    
    choice = st.sidebar.selectbox("Select Activities", activities)

    if choice == 'Classification Analysis Report':
        data = st.file_uploader("1. Upload a Dataset", type=["csv"])
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)

        print("{} {} ".format(seed_number, split_size))
        
        if data is not None:
            df = pd.read_csv(data)
            st.markdown('**1.1. Glimpse of dataset**')
            st.write(df)
            c.build_model(df, seed_number, split_size)

    if choice == 'ML Analysis Report':
        data = st.file_uploader("Upload a Dataset", type=["csv"])

        with st.sidebar.header('2. Set Parameters'):
            split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

        with st.sidebar.subheader('2.1. Learning Parameters'):
            parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
            parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
            parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
            parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

        with st.sidebar.subheader('2.2. General Parameters'):
            seed_number = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
            parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
            parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
            parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
            parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

        if data is not None:
            df = pd.read_csv(data)
            st.markdown('**1.1. Glimpse of dataset**')
            st.write(df)
            ml.build_model(df, seed_number, split_size)
        else:
            st.info('Awaiting CSV file to be uploaded.')
            # if st.button('Press to use Example Dataset'):
            #     # Code to handle using an example dataset
            #     boston = load_boston()
            #     X = pd.DataFrame(boston.data, columns=boston.feature_names).loc[:100]
            #     Y = pd.Series(boston.target, name='response').loc[:100]
            #     df = pd.concat([X, Y], axis=1)
            #     st.markdown('The Boston housing dataset is used as the example.')
            #     st.write(df.head(5))
            #     ml.build_model(df, parameter_n_estimators, parameter_max_features, parameter_min_samples_split, parameter_min_samples_leaf, seed_number, parameter_criterion, parameter_bootstrap, parameter_oob_score, parameter_n_jobs, split_size)

    if choice == 'General EDA':
        st.subheader("Exploratory Data Analysis")

        data = st.file_uploader("Upload a Dataset", type=["csv"])
        if data is not None:
            df = load.read_csv(data)
            st.dataframe(df.head())
            st.success("Data Frame Loaded successfully")

            if st.checkbox("Show dtypes"):
                st.write(dataframe.show_dtypes(df))

            if st.checkbox("Show Columns"):
                st.write(dataframe.show_columns(df))

            if st.checkbox("Show Missing"):
                st.write(dataframe.Show_Missing1(df))

            if st.checkbox("column information"):
                st.write(info.Column_information(df))

            if st.checkbox("Aggregation Tabulation"):
                st.write(dataframe.Tabulation(df))

            if st.checkbox("Num Count Summary"):
                st.write(info.num_count_summary(df))

            if st.checkbox("Statistical Summary"):
                st.write(info.statistical_summary(df))

            if st.checkbox("Show Selected Columns"):
                selected_columns = st.multiselect("Select Columns", dataframe.show_columns(df))
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Numerical Variables"):
                num_df = dataframe.Numerical_variables(df)
                numer_df = pd.DataFrame(num_df)
                st.dataframe(numer_df)

            if st.checkbox("Categorical Variables"):
                new_df = dataframe.categorical_variables(df)
                catego_df = pd.DataFrame(new_df)
                st.dataframe(catego_df)

            if st.checkbox("Replace null values with median"):
                selected_columns = st.multiselect("Select Columns", dataframe.show_columns(num_df))
                new_df = df[selected_columns]
                temp_df = dataframe.median(new_df)
                st.dataframe(temp_df)

            if st.checkbox("DropNA"):
                imp_df = dataframe.impute(num_df)
                st.dataframe(imp_df)

            if st.checkbox("Missing after DropNA"):
                st.write(dataframe.Show_Missing(imp_df))

            all_columns_names = dataframe.show_columns(df)
            all_columns_names1 = dataframe.show_columns(df)            
            selected_columns_names = st.selectbox("Select Column 1 For Cross Tabultion", all_columns_names)
            selected_columns_names1 = st.selectbox("Select Column 2 For Cross Tabultion", all_columns_names1)
            if st.button("Generate Cross Tab"):
                st.dataframe(pd.crosstab(df[selected_columns_names], df[selected_columns_names1]))

            all_columns_names3 = dataframe.show_columns(df)
            all_columns_names4 = dataframe.show_columns(df)            
            selected_columns_name3 = st.selectbox("Select Column 1 For Pearsonr Correlation (Numerical Columns)", all_columns_names3)
            selected_columns_names4 = st.selectbox("Select Column 2 For Pearsonr Correlation (Numerical Columns)", all_columns_names4)
            if st.button("Generate Pearsonr Correlation"):
                df = pd.DataFrame(dataframe.Show_pearsonr(imp_df[selected_columns_name3], imp_df[selected_columns_names4]), index=['Pvalue', '0'])
                st.dataframe(df)  

            

            st.subheader("UNIVARIATE ANALYSIS")
            
            all_columns_names = dataframe.show_columns(df)         
            selected_columns_names = st.selectbox("Select Column for Histogram ", all_columns_names)
            if st.checkbox("Show Histogram for Selected variable"):
                st.write(dataframe.show_hist(df[selected_columns_names]))
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()        

            all_columns_names = dataframe.show_columns(df)         
            selected_columns_names = st.selectbox("Select Columns Distplot ", all_columns_names)
            if st.checkbox("Show DisPlot for Selected variable"):
                st.write(dataframe.Show_DisPlot(df[selected_columns_names]))
                st.pyplot()

            all_columns_names = dataframe.show_columns(df)         
            selected_columns_names = st.selectbox("Select Columns CountPlot ", all_columns_names)
            if st.checkbox("Show CountPlot for Selected variable"):
                st.write(dataframe.Show_CountPlot(df[selected_columns_names]))
                st.pyplot()

            st.subheader("BIVARIATE ANALYSIS")

            Scatter1 = dataframe.show_columns(df)
            Scatter2 = dataframe.show_columns(df)            
            Scatter11 = st.selectbox("Select Column 1 For Scatter Plot (Numerical Columns)", Scatter1)
            Scatter22 = st.selectbox("Select Column 2 For Scatter Plot (Numerical Columns)", Scatter2)
            if st.button("Generate PLOTLY Scatter PLOT"):
                st.pyplot(dataframe.plotly(df, df[Scatter11], df[Scatter22]))
            
            bar1 = dataframe.show_columns(df)
            bar2 = dataframe.show_columns(df)            
            bar11 = st.selectbox("Select Column 1 For Bar Plot ", bar1)
            bar22 = st.selectbox("Select Column 2 For Bar Plot ", bar2)
            if st.button("Generate PLOTLY histogram PLOT"):
                st.pyplot(dataframe.plotly_histogram(df, df[bar11], df[bar22]))                

            violin1 = dataframe.show_columns(df)
            violin2 = dataframe.show_columns(df)            
            violin11 = st.selectbox("Select Column 1 For violin Plot", violin1)
            violin22 = st.selectbox("Select Column 2 For violin Plot", violin2)
            if st.button("Generate PLOTLY violin PLOT"):
                st.pyplot(dataframe.plotly_violin(df, df[violin11], df[violin22]))  

            st.subheader("MULTIVARIATE ANALYSIS")

            if st.checkbox("Show Histogram"):
                
                st.write(dataframe.show_hist(df))
                st.pyplot()

            if st.checkbox("Show HeatMap"):
                numerical_df = dataframe.Numerical_variables(df)
                st.write(dataframe.Show_HeatMap(df))
                st.pyplot()

            if st.checkbox("Show PairPlot"):
                st.write(dataframe.Show_PairPlot(df))
                st.pyplot()

            if st.button("Generate Word Cloud"):
                st.write(dataframe.wordcloud(df))
                st.pyplot()

    elif choice == 'EDA For Linear Models':
        st.subheader("EDA For Linear Models")
        data = st.file_uploader("Upload a Dataset", type=["csv"])
        if data is not None:
            df = load.read_csv(data)
            st.dataframe(df.head())
            st.success("Data Frame Loaded successfully")

            all_columns_names = dataframe.show_columns(df)         
            selected_columns_names = st.selectbox("Select Columns qqplot ", all_columns_names)
            if st.checkbox("Show qqplot for variable"):
                st.write(dataframe.qqplot(df[selected_columns_names]))
                st.pyplot()

            all_columns_names = dataframe.show_columns(df)         
            selected_columns_names = st.selectbox("Select Columns outlier ", all_columns_names)
            if st.checkbox("Show outliers in variable"):
                st.write(dataframe.outlier(df[selected_columns_names]))

            if st.checkbox("Show Distplot Selected Columns"):
                selected_columns_names = st.selectbox("Select Columns for Distplot ", all_columns_names)
                st.dataframe(dataframe.show_displot(df[selected_columns_names]))
                st.pyplot()

            con1 = dataframe.show_columns(df)
            con2 = dataframe.show_columns(df)            
            conn1 = st.selectbox("Select 1st Columns for chi square test", con1)
            conn2 = st.selectbox("Select 2st Columns for chi square test", con2)
            if st.button("Generate chi square test"):
                st.write(dataframe.check_cat_relation(df[conn1], df[conn2], 0.5))

    elif choice == 'Model Building for Classification Problem':
        st.subheader("Model Building for Classification Problem")
        data = st.file_uploader("Upload a Dataset", type=["csv"])
        if data is not None:
            df = load.read_csv(data)
            st.dataframe(df.head())
            st.success("Data Frame Loaded successfully")

            if st.checkbox("Select your Variables  (Target Variable should be at last)"):
                selected_columns_ = st.multiselect("Select Columns for seperation ", dataframe.show_columns(df))
                sep_df = df[selected_columns_]
                st.dataframe(sep_df)

            if st.checkbox("Show Independent Data"):
                x = sep_df.iloc[:, :-1]
                st.dataframe(x)

            if st.checkbox("Show Dependent Data"):
                y = sep_df.iloc[:, -1]
                st.dataframe(y)

            if st.checkbox("Dummay Variable"):
                x = dataframe.dummy(x)
                st.dataframe(x)

            if st.checkbox("IMputer "):
                x = model.IMpupter(x)
                st.dataframe(x)

            if st.checkbox("Compute Principle Component Analysis"):
                x = dataframe.PCA(x)
                st.dataframe(x)

            st.subheader("TRAIN TEST SPLIT")

            if st.checkbox("Select X Train"):
                from sklearn.model_selection import train_test_split
                x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
                st.dataframe(x_train)

            if st.checkbox("Select x_test"):
                from sklearn.model_selection import train_test_split
                x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
                st.dataframe(x_test)

            if st.checkbox("Select y_train"):
                from sklearn.model_selection import train_test_split
                x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
                st.dataframe(y_train)

            if st.checkbox("Select y_test"):
                from sklearn.model_selection import train_test_split
                x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
                st.dataframe(y_test)

            st.subheader("MODEL BUILDING")
            st.write("Build your BaseLine Model")

            if st.checkbox("Logistic Regression "):
                x = model.Logistic_Regression(x_train, y_train, x_test, y_test)
                st.write(x)

            if st.checkbox("Decision Tree "):
                x = model.Decision_Tree(x_train, y_train, x_test, y_test)
                st.write(x)

            if st.checkbox("Random Forest "):
                x = model.RandomForest(x_train, y_train, x_test, y_test)
                st.write(x)

            if st.checkbox("naive_bayes "):
                x = model.naive_bayes(x_train, y_train, x_test, y_test)
                st.write(x)

            if st.checkbox("XGB Classifier "):
                x = model.XGb_classifier(x_train, y_train, x_test, y_test)
                st.write(x)

    st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)


if __name__ == '__main__':
    load = DataFrame_Loader()
    dataframe = EDA_Dataframe_Analysis()
    info = Attribute_Information()
    model = Data_Base_Modelling()
    ml = ML_Analyses()
    c = Classification_Analyses()
    main()
