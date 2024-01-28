import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

class dataproc:
    @staticmethod
    def negative_deal(df):
    q2 = df.quantile(0.5)
    for i in df:
        if i <=0 :
            df.replace(i,q2,inplace = True)
    return df
    @staticmethod
    def impute_missing_values(df):
        # Extract the target column
        target = df['编号']

        # Select relevant features for regression
        features = df.loc[:, ['性别', '年龄', '体重指数', '糖尿病家族史', '舒张压', '口服耐糖量测试', '胰岛素释放实验', '肱三头肌皮褶厚度']]

        # Filter rows with missing values
        X_missing_reg = features[~features.isin([0])]

        # Calculate missing value ratio
        missing = missing[~missing['缺失值个数'].isin([0])]
        missing['缺失比例'] = missing['缺失值个数'] / X_missing_reg.shape[0]

        # Calculate missing values in each column
        X_df = X_missing_reg.isnull().sum()

        # Sort columns by the number of missing values
        colname = X_df[~X_df.isin([0])].sort_values().index.values

        # Sort columns based on missing values from least to most
        sortindex = [X_missing_reg.columns.tolist().index(str(i)) for i in colname]

        for i in sortindex:
            # Build new feature matrix and label
            df = X_missing_reg
            fillc = df.iloc[:, i]

            # Concatenate non-missing columns and the original complete label to create a new feature matrix
            df = pd.concat([df.drop(df.columns[i], axis=1), pd.DataFrame(y_full)], axis=1)

            # Impute missing values with 0 in the new feature matrix
            df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)

            # Split into training and testing sets
            Ytrain = fillc[fillc.notnull()]
            Ytest = fillc[fillc.isnull()]
            Xtrain = df_0[Ytrain.index, :]
            Xtest = df_0[Ytest.index, :]

            # Train a Random Forest Regressor
            rfc = RandomForestRegressor(n_estimators=100)
            rfc = rfc.fit(Xtrain, Ytrain)
            Ypredict = rfc.predict(Xtest)

            # Fill the missing values in the original feature matrix with predicted values
            X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), X_missing_reg.columns[i]] = Ypredict

        df = X_missing_reg

        return df
        
        @staticmethod
        def BMI(x):
            if x < 18.5:
                return 0
            elif 18.5 <= x <=23.9:
                return 1
            elif 24 <= x <= 27.9:
                return 2
            elif 28 <= x <= 29.9:
                return 3
            elif x >= 30:
                return 4
        @staticmethod
        def DP(x):
            if x < 80:
                return 0
            elif 80 <= x < 89:
                return 1
            elif 89 <= x < 99:
                return 2
            elif 99 <= x < 109:
                return 3
            elif x >=109:
                return 4
        @staticmethod
        def OGTT(x):
            if x <= 7.8:
                return 0
            else:
                return 1
        @staticmethod
        def load_and_clean_data():
            #Loading Data
            tnb = pd.read_csv(r"../input/trainkdxf/train.csv",encoding='gbk')

            print(tnb.shape)	# Dimension
            name=tnb.columns.tolist() # Column name
            #taking out the predictor
            tnbdata=tnb.loc[:,['性别', '出生年份', '体重指数', '糖尿病家族史', '舒张压', '口服耐糖量测试', '胰岛素释放实验',
                   '肱三头肌皮褶厚度']]

            tnbdata.loc[(tnbdata['糖尿病家族史'])=='叔叔或者姑姑有一方患有糖尿病','糖尿病家族史']=1
            tnbdata.loc[(tnbdata['糖尿病家族史'])=='叔叔或姑姑有一方患有糖尿病','糖尿病家族史']=1
            tnbdata.loc[(tnbdata['糖尿病家族史'])=='无记录','糖尿病家族史']=0
            tnbdata.loc[(tnbdata['糖尿病家族史'])=='父母有一方患有糖尿病','糖尿病家族史']=2

            tnbdata['糖尿病家族史']=tnbdata['糖尿病家族史'].astype('int')
            tnbdata['出生年份']=2022-tnbdata['出生年份'][:]
            tnbdata.rename(columns={
                '出生年份': '年龄'}, inplace=True)
            tnbdata = impute_missing_values(tnbdata)
            negative_list = ['体重指数','肱三头肌皮褶厚度','胰岛素释放实验','口服耐糖量测试']
            for i in negative_list:
                tnbdata[i] = negative_deal(tnbdata[i])
            tnbdata['体质指数-BMI'] = tnbdata['体重指数'].map(BMI)
            tnbdata['舒张压-DP'] = tnbdata['舒张压'].map(DP)
            tnbdata['舒张压-DP'] = tnbdata['舒张压'].map(OGTT)


            return tnbdata

    
    
    @staticmethod
    def valid_dataset():
        finaltest=pd.read_csv(r"../input/finaltest5/testB.csv",encoding='utf-8')
        finaltest['舒张压']=pd.to_numeric(finaltest['舒张压'])
        #making unit in the two dataset consistent
        finaltest['肱三头肌皮褶厚度']=finaltest['肱三头肌皮褶厚度']*10
        
        finaltest.loc[(finaltest['糖尿病家族史'])=='叔叔或者姑姑有一方患有糖尿病','糖尿病家族史']=1
        finaltest.loc[(finaltest['糖尿病家族史'])=='叔叔或姑姑有一方患有糖尿病','糖尿病家族史']=1
        finaltest.loc[(finaltest['糖尿病家族史'])=='无记录','糖尿病家族史']=0
        finaltest.loc[(finaltest['糖尿病家族史'])=='父母有一方患有糖尿病','糖尿病家族史']=2
        
        finaltest['糖尿病家族史']=finaltest['糖尿病家族史'].astype('int')
        finaltest['出生年份']=2022-finaltest['出生年份'][:]
        finaltest.rename(columns={
            '出生年份': '年龄'}, inplace=True)      
        finaltest = impute_missing_values(finaltest)
        negative_list = ['体重指数','肱三头肌皮褶厚度','胰岛素释放实验','口服耐糖量测试']
                    for i in negative_list:
                        tnbdata[i] = negative_deal(tnbdata[i])
        finaltest['体质指数-BMI'] = finaltest['体重指数'].map(BMI)
        finaltest['舒张压-DP'] = finaltest['舒张压'].map(DP)
        finaltest['舒张压-DP'] = finaltest['舒张压'].map(OGTT)

        return finaltest

