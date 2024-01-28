import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import graphviz 

class selection:
    def perform_feature_selection(self):
        tnb = pd.read_csv(r"../input/trainkdxf/train.csv",encoding='gbk')
        # Shuffle the data and split it into training and testing sets
        x, y=su.shuffle(tnbdata, tnb['患有糖尿病标识'], random_state=7)
        train_size = int(len(x)*0.8)
        train_x, test_x, train_y, test_y =             x[:train_size],x[train_size:],             y[:train_size], y[train_size:]
        # Create a decision tree classifier model, train it on the training set, and test it on the testing set
        t_model = st.DecisionTreeClassifier(max_depth=6)
        t_model=st.DecisionTreeClassifier(max_depth=6)
        # Use AdaBoost to create an ensemble of decision trees
        model_ad = se.AdaBoostClassifier(n_estimators=100, random_state=7)
        model_ad.fit(train_x, train_y)
        pred_test_y = model_ad.predict(test_x)
        #print(sm.r2_score(test_y, pred_test_y))
        # Calculate feature importances for AdaBoost
        fi_ab = model_ad.feature_importances_
        #print(fi_ab)
        # Define feature names
        name=['性别','年龄', '体重指数','糖尿病家族史','舒张压', '口服耐糖量测试', '胰岛素释放实验',
           '肱三头肌皮褶厚度','体质指数-BMI','舒张压-DP','口服耐糖量测试-OGTT']
        name=np.array(name)
        # Train a decision tree model on the training set and test it on the testing set
        model = st.DecisionTreeClassifier(max_depth=6)
        model.fit(train_x, train_y)
        pred_test_y = model.predict(test_x)
        #print(sm.r2_score(test_y, pred_test_y))
        # Calculate feature importances for the decision tree
        fi_t = model.feature_importances_
        dot_data = st.export_graphviz(model) 
        graph = graphviz.Source(dot_data)  
        graph 
        # Visualize feature importances using bar plots
        plt.figure('feature_importances',facecolor='lightgray',figsize=(15,15),dpi=80)
        plt.rcParams['font.sans-serif']=['Arial Unicode MS']
        plt.subplot(211)
        plt.title('feature_importances',fontsize=16)
        plt.ylabel('Feature Importances', fontsize=12)
        plt.grid(linestyle=':')
        sorted_indices = fi_ab.argsort()[::-1]
        x = np.arange(fi_ab.size)
        plt.xticks(x, name[sorted_indices])
        plt.bar(x, fi_ab[sorted_indices], color='dodgerblue', label='fi_ab')
        plt.legend()

        plt.subplot(212)
        plt.title('feature_importances',fontsize=16)
        plt.ylabel('Feature Importances', fontsize=12)
        plt.grid(linestyle=':')
        sorted_indices = fi_t.argsort()[::-1]
        x = np.arange(fi_t.size)
        plt.xticks(x,name[sorted_indices])
        plt.bar(x, fi_t[sorted_indices], color='orangered', label='fi_t')
        plt.legend()

        plt.show()

