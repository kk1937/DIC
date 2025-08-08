# -*- coding: gbk -*-
from calendar import c
import datetime
from ensurepip import bootstrap
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error,roc_curve, auc
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.utils import resample
from sklearn.metrics import matthews_corrcoef, brier_score_loss,make_scorer,precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import calibration_curve
import shap
import pandas as pd
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import ttest_ind
import statsmodels.api as sm

def bootstrap_confidence_interval(y_true, y_scores, y_pred, metric_func, n_bootstrap=1000):
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        y_true_resample, y_scores_resample, y_pred_resample = resample(y_true, y_scores, y_pred)
        bootstrap_sample = metric_func(y_true_resample, y_scores_resample, y_pred_resample)
        bootstrap_samples.append(bootstrap_sample)

    bootstrap_samples = np.array(bootstrap_samples)
    lower = np.percentile(bootstrap_samples, 2.5)
    upper = np.percentile(bootstrap_samples, 97.5)

    return lower, upper

def calculate_metrics(model, X_Data, y_Data, name):
    # Initialize a dictionary to store metric values for each metric
    metrics_samples = {
        'auroc': [],
        'pr_auc': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f_score': [],
        'cohen_kappa': []  # ���� Cohen's Kappa
    }

    # Perform 100 bootstrap iterations
    for _ in range(100):
        # Resample the data for each iteration
        X_resample, y_resample = resample(X_Data, y_Data)

        # Get predictions and predicted probabilities
        y_scores = model.predict_proba(X_resample)[:, 1]
        y_pred = model.predict(X_resample)

        # Calculate metrics
        precision, recall, _ = metrics.precision_recall_curve(y_resample, y_scores)
        sort_idx = np.argsort(recall)
        precision = precision[sort_idx]
        recall = recall[sort_idx]
        if len(set(y_resample)) > 1:
            metrics_samples['auroc'].append(metrics.roc_auc_score(y_resample, y_scores))
        else: 
            metrics_samples['auroc'].append(0)
        metrics_samples['pr_auc'].append(metrics.auc(recall, precision))
        metrics_samples['accuracy'].append(metrics.accuracy_score(y_resample, y_pred))
        metrics_samples['sensitivity'].append(metrics.recall_score(y_resample, y_pred))
        metrics_samples['specificity'].append(metrics.recall_score(1-y_resample, 1-y_pred))
        metrics_samples['precision'].append(metrics.precision_score(y_resample, y_pred, zero_division=1))
        metrics_samples['f_score'].append(metrics.f1_score(y_resample, y_pred))
        metrics_samples['cohen_kappa'].append(metrics.cohen_kappa_score(y_resample, y_pred))  # ���� Cohen's Kappa

    # Calculate 95% confidence interval for each metric
    metrics_dict = {}
    for metric, samples in metrics_samples.items():
        lower = np.percentile(samples, 2.5)
        upper = np.percentile(samples, 97.5)
        metrics_dict[f'{metric}'] = np.mean(samples)
        metrics_dict[f'{metric}_lower'] = lower
        metrics_dict[f'{metric}_upper'] = upper

    # Return as DataFrame
    df = pd.DataFrame(metrics_dict, index=[name])
    return df


    # Calculate MCC curve
def calculate_mcc_curve(y_val_prob, X_val, y_val):
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
    mcc_scores = [matthews_corrcoef(y_val, y_val_prob >= t) for t in thresholds]
    return thresholds, mcc_scores
    #
def plot_mcc_curves(results, save_path):
    plt.figure(figsize=(10, 6))
    
    for thresholds, mcc_scores, label in results:
        # Find the maximum MCC value and its corresponding threshold
        max_mcc = max(mcc_scores)
        max_threshold = thresholds[mcc_scores.index(max_mcc)]
        
        # Optionally, plot with threshold annotation
        plt.plot(thresholds, mcc_scores, label=f'{label} (Max MCC: {max_mcc:.2f})')
    
    plt.xlabel('Threshold')
    plt.ylabel('MCC')
    plt.title('MCC Curves for Different Models')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()


def plot_precision_recall_curves(pr_curves, title, save_path):
    # Create a new figure
    plt.figure(figsize=(8, 6))

    # Plot Precision-Recall curve for each model
    for i, (precision, recall, name,score) in enumerate(pr_curves):
        plt.plot(recall, precision, label=f'{name} ({score:.2f})')

    # Set axis labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")

    # Save the figure
    plt.savefig(save_path)
    plt.close()

def plot_calibration_curves(calibration_data, title, save_path):
    # Create a new figure
    plt.figure(figsize=(8, 6))

    # Plot calibration curve for each model
    for fraction_of_positives, mean_predicted_value, name,score in calibration_data:
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{name} (Brier: {score:.2f})')

    # Plot the ideal calibration line
    plt.plot([0, 1], [0, 1], '--', color='gray')

    # Set axis labels and title
    plt.ylabel('Fraction of positives')
    plt.xlabel('Mean predicted value')
    plt.title(title)
    plt.legend(loc="lower right")

    # Save the figure
    plt.savefig(save_path)
    plt.close()

def calculate_dic(row):
    # Calculate PLT score
    if 80 < row['Post-PLT'] <= 120:
        plt_score = 1
    elif 50 < row['Post-PLT'] <= 80:
        plt_score = 2
    elif 0 < row['Post-PLT'] <= 50:
        plt_score = 3
    else:
        plt_score = 0

    # Calculate PT score
    if 13.5 <= row['Post-PT'] < 19.5:
        pt_score = 1
    elif row['Post-PT'] >= 19.5:
        pt_score = 2
    else:
        pt_score = 0

    # Calculate INR score
    if 1.25 <= row['Post-INR'] < 1.5:
        inr_score = 1
    elif row['Post-INR'] >= 1.5:
        inr_score = 2
    else:
        inr_score = 0

    # Calculate APTT score
    if row['Post-APTT'] >= 58:
        aptt_score = 1
    else:
        aptt_score = 0

    # Calculate DIC total score
    dic_score = plt_score + pt_score + inr_score + aptt_score
    if dic_score >= 4:
        return 1
    else:
        return 0

plt.rcParams['font.sans-serif'] = ['SimHei']  # Set default font
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is displayed correctly
labelCol='DIC'

 # 1. Data loading
 #region  load data

loadedData=0
if loadedData==1:
    path=str(r"./DIC/Data/output/filled_data1.csv")
    allData = pd.read_csv(path,encoding='UTF-8') 
    # Filter sources 0, 1, 2, 3 and print DIC positive rate for each source
    filtered_data = allData[allData['source'].isin([0, 1, 2, 3])]
    for source in [0, 1, 2, 3]:
        source_data = filtered_data[filtered_data['source'] == source]
        positive_count = source_data['DIC'].sum()
        total_count = source_data.shape[0]
        positive_rate = positive_count / total_count if total_count > 0 else 0
        print(f'Source {source} total: {total_count}, DIC positive rate: {positive_rate:.2%}, positive count: {positive_count}')
else:
    path=str(r"./DIC/Data/��һ�������ı�1228.csv")
    data = pd.read_csv(path,encoding='UTF-8')
    

    # Calculate DIC for each row in data
    data['DIC'] = data.apply(calculate_dic, axis=1)
    
    data.to_csv(r'./DIC/Data/output/added DIC.csv', index=False,encoding='utf-8')
    # Remove rows where 'Pre-PLT' is missing (if needed)
 
    #data = data.dropna(subset=['Pre-WBC'])
    #data = data.dropna(subset=['Pre-PLT'])
    
    data = data[(data['ISS'] >= 16) | (data['DIC'] == 1)]
    data.to_csv(r'./DIC/Data/output/ISSgt16 or DICis1.csv', index=False,encoding='utf-8')
    #data = data.dropna(subset=['Admission SBP'])
    # 1.5 Data analysis: logistic regression for DIC and other indicators
    print(f'-----------1.5 Data Analysis')
    import statsmodels.api as sm
    dic_col = 'DIC'
    indicator_info = pd.read_csv('./DIC/Data/indicators.csv', encoding='utf-8')
    indicator_cols = indicator_info[indicator_info['����'] == 1]['����'].tolist()
    results = {}
    for col in indicator_cols:
        X = data[[dic_col]]
        y = data[col]
        X = sm.add_constant(X)
        mask = y.isin([0, 1])
        X_valid = X[mask]
        y_valid = y[mask]
        X_valid = X_valid.astype(float)
        y_valid = y_valid.astype(int)
        model = sm.Logit(y_valid, X_valid).fit(disp=0)
        results[col] = model.summary2().tables[1]
    for col, summary in results.items():
        print(f"\nLogistic regression summary for {col}")
        print(summary)
    import numpy as np
    import pandas as pd
    or_list = []
    ci_lower = []
    ci_upper = []
    p_values = []
    names = []
    for col, summary in results.items():
        coef = summary.loc['DIC', 'Coef.']
        se = summary.loc['DIC', 'Std.Err.']
        or_val = np.exp(coef)
        ci_l = np.exp(coef - 1.96 * se)
        ci_u = np.exp(coef + 1.96 * se)
        p = summary.loc['DIC', 'P>|z|']
        or_list.append(or_val)
        ci_lower.append(ci_l)
        ci_upper.append(ci_u)
        p_values.append(p)
        names.append(col)
    df_plot = pd.DataFrame({
        'Indicator': names,
        'OR': or_list,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'p_value': p_values
    })
    df_plot.to_csv('./DIC/Data/df_plot.csv', index=False, encoding='utf-8')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, len(df_plot)*0.6))
    plt.errorbar(df_plot['OR'], df_plot['Indicator'], 
                 xerr=[df_plot['OR']-df_plot['CI_lower'], df_plot['CI_upper']-df_plot['OR']],
                 fmt='o', color='blue', ecolor='gray', capsize=4)
    plt.axvline(1, color='red', linestyle='--')
    plt.xlabel('OR (95% CI)')
    plt.title('DIC Logistic Regression OR and 95% CI for Each Indicator')
    plt.tight_layout()
    plt.savefig('./DIC/Data/dic_logistic_forest.png', dpi=300)
    data = data.dropna(subset=['Pre-APTT'])
    #data = data.dropna(subset=['Injury mechanism'])
    # ɾ�� source ��Ϊ 0 �� Date of injury ��Ϊ�յ���
    #zenmewomen dwo zajiejudata = data[~((data['source'] == 0) & (data['Date of injury'].isna()))]

    
    # Print DIC positive rate for each source
    filtered_data = data
    for source in [0, 1, 2, 3]:
        source_data = filtered_data[filtered_data['source'] == source]
        positive_count = source_data['DIC'].sum()
        total_count = source_data.shape[0]
        positive_rate = positive_count / total_count if total_count > 0 else 0
        print(f'Source {source} total: {total_count}, DIC positive rate: {positive_rate:.2%}, positive count: {positive_count}')
    
    data= data.dropna(subset=['Pre-PLT'])


    # Print DIC positive rate for each source (filtered)
    filtered_data = data
    for source in [0, 1, 2, 3]:
        source_data = filtered_data[filtered_data['source'] == source]
        positive_count = source_data['DIC'].sum()
        total_count = source_data.shape[0]
        positive_rate = positive_count / total_count if total_count > 0 else 0
        print(f'Filtered Source {source} DIC positive rate: {positive_rate:.2%}, positive count: {positive_count}')
    feature_vars=pd.read_csv(str(r"./DIC/Data/Death��������.csv"),encoding='UTF-8')
    # Select features where type is not 0
    print(f'Selecting features where type is not 0')
    selected_rows = feature_vars[feature_vars['����'] != 0]
    # Get the feature names
    selected_columns = selected_rows['����'].str.strip()
    usedColumns=selected_columns.tolist()
    if 'source' not in usedColumns:
        usedColumns.append('source')
    if 'year_group' not in usedColumns:
        usedColumns.append('year_group')
    
    if 'Admission time' not in usedColumns:
        usedColumns.append('Admission time')
    if 'DIC' not in usedColumns:
        usedColumns.append('DIC')
    data=data[usedColumns]
    
    # Remove duplicated columns
    duplicated_columns = data.columns[data.columns.duplicated()]
    print(f"Duplicated columns: {duplicated_columns.tolist()}")
    data = data.loc[:, ~data.columns.duplicated()]

    for col in data.columns:
        if (col!='Surgery start time') & (col!='Admission time'):
            data[col] = pd.to_numeric(data[col], errors='coerce')
    continuous_vars = []
    categorical_vars = []
    binary_vars = []
    # Classify each column
    for col in data.columns:
        if (col=='Surgery start time') or (col=='Admission time') :
            continue
        if data[col].nunique() < 4:
            binary_vars.append(col)
        # If unique values < 10, treat as categorical, else as continuous
        else:
            if data[col].nunique() < 10:
                categorical_vars.append(col)
            else:
                continuous_vars.append(col)
        
    # Print missing value count for each column
    print(f'Count missing values for each column')        
    missing_ratio = data.isnull().sum() / len(data)
    # Find columns with missing ratio > 0.462
    columns_to_drop = missing_ratio[(missing_ratio > 0.462)& (missing_ratio.index != 'Admission time') & (missing_ratio.index != 'year_group')].index
    # Drop these columns
    print(f'Drop columns with missing ratio > 0.42: {len(columns_to_drop)}')
    
    columns_to_dropRate = missing_ratio[missing_ratio > 0.462]
    
        print(f'����: {column}, ����: {var_type}, ��ȱʧ����: {non_null_count}, ȱʧ����: {ratio:.2%}')

    data = data.drop(columns_to_drop, axis=1)

    # Fill missing values
    df = data
    df[binary_vars] = df[binary_vars].fillna(0)
    # For continuous and categorical variables, use KNNImputer
    print(f'Imputing missing values for continuous and categorical variables using KNNImputer')
    imputer = KNNImputer(n_neighbors=5)
    df[continuous_vars] = imputer.fit_transform(df[continuous_vars])
    if len(categorical_vars) > 0:
        # For categorical variables, encode to ordinal, impute, then decode
        encoder = OrdinalEncoder()
        df[categorical_vars] = encoder.fit_transform(df[categorical_vars])
        df[categorical_vars] = imputer.fit_transform(df[categorical_vars])
        df[categorical_vars] = encoder.inverse_transform(df[categorical_vars])
    allData = df
    allData.to_csv(r'./DIC/Data/output/filled_data1.csv', index=False, encoding='utf-8')

#endregion load data

  
# 2.����ѵ�����Ͳ��Լ�
#region  split data

    
print(f'-----------2.����ѵ�����Ͳ��Լ�')
#outSideData = allData[(allData['source'] == 1) | (allData['source'] == 0) | (allData['source'] == 2)]
# Handle NaN values if present
allData['Admission time'] = allData['Admission time'].fillna('')

# Ensure 'Admission time' column is of string type
allData['Admission time'] = allData['Admission time'].astype(str)

# ɸѡ�� inSideData��������('source'==3 ��'year_group'<2017) ���� ('source'!=3 ��'Admission time'��ǰ4λ<2021)
# ���Խ� 'Admission time' ��ת��Ϊʱ���ʽ
def extract_year(admission_time):
    try:
        ddd=allData['Admission time']
        return pd.to_datetime(admission_time).year
    except ValueError:
        return int(admission_time[:4])

# Ӧ�õ� 'Admission time' ��
allData['Admission time'] = allData['Admission time'].apply(extract_year)


inSideData = allData[allData['source'].isin([0,1,2])]
# �������ݷŵ� outSideData ��
outSideData = allData[~allData.index.isin(inSideData.index)]


positive_count = inSideData['DIC'].sum()
total_count = inSideData.shape[0]
positive_rate = positive_count / total_count if total_count > 0 else 0
print(f'inSideData  ��������{total_count}�� DIC ������: {positive_rate:.2%}, ��������: {positive_count}')

positive_count = outSideData['DIC'].sum()
total_count = outSideData.shape[0]
positive_rate = positive_count / total_count if total_count > 0 else 0
print(f'outSideData  ��������{total_count}�� DIC ������: {positive_rate:.2%}, ��������: {positive_count}')    

for source in [0, 1,2, 3]:
    source_data = inSideData[inSideData['source'] == source]
    positive_count = source_data['DIC'].sum()
    total_count = source_data.shape[0]
    positive_rate = positive_count / total_count if total_count > 0 else 0
    print(f'inSideData: Source {source}  ��������{total_count}�� DIC ������: {positive_rate:.2%}, ��������: {positive_count}')
for source in [0, 1,2, 3]:
    source_data = outSideData[outSideData['source'] == source]
    positive_count = source_data['DIC'].sum()
    total_count = source_data.shape[0]
    positive_rate = positive_count / total_count if total_count > 0 else 0
    print(f'outSideData: Source {source}  ��������{total_count}�� DIC ������: {positive_rate:.2%}, ��������: {positive_count}')

#outSideData = allData[(allData['source'] ==1 )| (allData['source'] == 0) ]
# ���� inSideData �е�����˳��
inSideData = inSideData.sample(frac=1).reset_index(drop=True)
inSideData = inSideData.sample(frac=1).reset_index(drop=True)

# ʹ�� groupby �� describe ������ȡͳ����Ϣ
grouped_stats = allData.groupby('source').describe()

# ��ӡͳ����Ϣ
print(grouped_stats)
grouped_stats.to_csv(r'./DIC/Data/output/grouped_stats.csv')
#Pcount = sum(data[labelCol] == 1)
outSideData= outSideData.drop('source', axis=1)
# outSideData= outSideData.drop('Surgery start time', axis=1)
inSideData= inSideData.drop('source', axis=1)
# inSideData= inSideData.drop('Surgery start time', axis=1)
X = inSideData.drop(labelCol, axis=1)
y = inSideData[labelCol]

X_outTest=outSideData.drop(labelCol, axis=1)
y_outTest=outSideData[labelCol]

X_All=allData.drop('source', axis=1).drop(labelCol, axis=1)
y_All=allData.drop('source', axis=1)[labelCol]
# outSideTestData=outSideData
# X_outTest=outSideTestData.drop(labelCol, axis=1)
# y_outTest=outSideTestData[labelCol]

X_train, X_insidetest, y_train, y_insidetest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle= False)

#endregion split data

# 3. Over/Under Sampling
#region SMOTE
#-------------Start SMOTE----------------
# # Use SMOTE method
print(f'-----------3. Over/Under Sampling')
# # from imblearn.over_sampling import SMOTE

outTestRate= y_outTest.sum()/len(y_outTest)
#outTestRate=0.5
print(f'outTestRate:{outTestRate}')
trainRate= y.sum()/len(y)
print(f'trainTestRate:{trainRate}')
useSmote=outTestRate>trainRate
targetNumber=int(len(y_train) * outTestRate)
print(f'targetNumber:{targetNumber}')

if(useSmote):
    smote = SMOTE(sampling_strategy={1:targetNumber}, random_state=42)
    X, y = smote.fit_resample(X, y)
else:
    undersample = RandomUnderSampler(sampling_strategy={1: targetNumber})
    X, y = undersample.fit_resample(X, y)
    

X_train, X_insidetest, y_train, y_insidetest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#endregion



# 4. Feature Selection
#region feature selection

print(f'-----------4. Feature Selection')


from scipy.stats import ks_2samp
import pandas as pd

 # Compare X_train and X_test using Pandas DataFrame
ks_results = []

 # Perform KS test for each feature
for column in X_train.columns:
    ks_stat, p_value = ks_2samp(X_train[column], X_outTest[column])
    ks_results.append({'Feature': column, 'KS Statistic': ks_stat, 'P-Value': p_value})
    print(f"Feature: {column}, KS Statistic: {ks_stat:.4f}, P-Value: {p_value:.4f}")

 # Convert to DataFrame
ks_results_df = pd.DataFrame(ks_results)

 # Bonferroni correction
alpha = 0.05  # Original significance level
m = len(ks_results_df)  # Total number of features
alpha_adjusted = alpha / m  # Adjusted significance level

 # Select features with p-value < alpha_adjusted
ks_results_df['Bonferroni_Significant'] = ks_results_df['P-Value'] > alpha_adjusted
significant_features = ks_results_df[ks_results_df['Bonferroni_Significant']]['Feature'].tolist()

 # print(f"Removed features: {significant_features}")

correlations = X_All.corrwith(y_All)

 # Select top 16 features with highest correlation
top_50_features = correlations.abs().nlargest(16).index
# Convert Index to list
top_50_features_list = list(top_50_features)

# Remove the element
if 'Admission time' in top_50_features_list:
    top_50_features_list.remove('Admission time')

# Convert list back to Index if necessary
top_50_features = pd.Index(top_50_features_list)
X_top_50 = X_All[top_50_features]
selected_features=X_top_50.columns


print(f"{datetime.datetime.now()}-- ʹ�� LassoCV ���н�����֤��ѡ�����ŵ� alpha ����")
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_top_50, y_All)

# ��ȡ���ŵ� alpha ֵ
best_alpha = lasso_cv.alpha_
print(f"{datetime.datetime.now()}-- ���ŵ� alpha ֵ: {best_alpha}")

# ʹ�����ŵ� alpha ֵ����ѵ�� LASSO ģ��
lasso = Lasso(alpha=best_alpha)
lasso.fit(X_top_50, y_All)

# ʹ�� SelectFromModel ѡ������
model = SelectFromModel(lasso, prefit=True)

# ��ȡѡ�����������
selected_features = X_top_50.columns[model.get_support()]

print("����ѡ���������", selected_features)
# # ʹ��LASSO�ع��������ѡ��
lasso = Lasso(alpha=0.01)
lasso.fit(X_top_50, y_All)

# # ʹ��SelectFromModelѡ������
model = SelectFromModel(lasso, max_features=15, prefit=True)
selected_features = X_top_50.columns[model.get_support()]

#selected_features=selected_features.union(significant_features)
print("Selected features:", selected_features)


X= X[selected_features]
X_train = X_train[selected_features]
X_insidetest= X_insidetest[selected_features]
X_outTest = X_outTest[selected_features]

ttest_results = []
for feature in selected_features:
    group1 = allData[allData['DIC'] == 1][feature]
    group0 = allData[allData['DIC'] == 0][feature]
   
    group1 = pd.Series(group1).dropna().values
    group0 = pd.Series(group0).dropna().values
    mean1, std1 = group1.mean(), group1.std()
    mean0, std0 = group0.mean(), group0.std()
    t_stat, p_val = ttest_ind(group1, group0, equal_var=False)
    ttest_results.append({
        'Feature': feature,
        'T-statistic': t_stat,
        'P-value': p_val,
        'DIC=1 Mean��SD': f'{mean1:.2f}��{std1:.2f}',
        'DIC=0 Mean��SD': f'{mean0:.2f}��{std0:.2f}'
    })

ttest_df = pd.DataFrame(ttest_results)
ttest_df.to_csv('./DIC/Data/output/ttest_results.csv', index=False, encoding='utf-8')
print('T�������ѱ���Ϊ selected_features_ttest_results.csv')

#endregion feature selection

# 5. Standardize training and test data
#region  standardization
print(f'-----------5.��ѵ�����ݽ��б�׼��')
# Create a standard scaler
scaler = StandardScaler()
X_train_original = X_train.copy()
# Standardize training data
X_train = scaler.fit_transform(X_train)

# Use the same scaler to standardize test data
X_insidetest=scaler.transform(X_insidetest)
X_outTest =scaler.transform(X_outTest)

# Convert X to numpy array
X_train = np.array(X_train)

# Calculate Z-score
z_scores = np.abs(stats.zscore(X_train))

# Set a threshold, usually 3, meaning points with Z-score > 3 are considered outliers
threshold = 3

# Get outlier indices
outliers = np.where(z_scores > threshold)

# Replace outlier values with threshold
X_train[outliers] = threshold


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Combine X_train and X_outTest for PCA visualization
# Assign labels for train and test
X_combined = np.vstack((X_train, X_outTest))
y_combined = np.hstack((np.zeros(len(X_train)), np.ones(len(X_outTest))))  # 0 ��ʾѵ������1 ��ʾ���Լ�

# Use PCA to reduce data to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)
# Split train and test after PCA
X_train_pca = X_pca[y_combined == 0]
X_test_pca = X_pca[y_combined == 1]
# Calculate centroids for each class
centroids = {
    label: np.mean(X_pca[y_combined == label], axis=0)
    for label in np.unique(y_combined)
}

# Calculate distance from each point to its class centroid
distances = np.array([
    np.linalg.norm(X_pca[i] - centroids[y_combined[i]])
    for i in range(len(X_pca))
])

# Set outlier threshold at 95th percentile
threshold = np.percentile(distances, 95)

# Identify outliers
outliers = distances > threshold

# Remove outliers from X_outTest and y_outTest
X_outTest = X_outTest[~outliers[len(X_train):]]
y_outTest = y_outTest[~outliers[len(X_train):]]
outlier_indices = np.where(outliers)[0]
outlier_points = X_pca[outlier_indices]
# Plot visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], label='Train', alpha=0.5, c='blue')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], label='Test', alpha=0.5, c='orange')

plt.scatter(outlier_points[:, 0], outlier_points[:, 1], label='Outliers', alpha=0.8, c='red', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization with Outliers')
plt.legend()
plt.savefig('./DIC/Data/output/pca_visualization.png', dpi=300, bbox_inches='tight')  # dpi=300 ensures high resolution
print("Figure saved as pca_visualization.png")

# Repeat the process
scaler = StandardScaler()
X_train_original = X_train.copy()
# Standardize training data again
X_train = scaler.fit_transform(X_train)

# Use the same scaler to standardize test data again
X_insidetest=scaler.transform(X_insidetest)
X_outTest =scaler.transform(X_outTest)

# Convert X to numpy array again
X_train = np.array(X_train)

# Calculate Z-score again
z_scores = np.abs(stats.zscore(X_train))

# Set threshold for outliers again
threshold = 3

# Get outlier indices again
outliers = np.where(z_scores > threshold)

# Replace outlier values with threshold again
X_train[outliers] = threshold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Combine X_train and X_outTest for PCA visualization again
# Assign labels for train and test again
X_combined = np.vstack((X_train, X_outTest))
y_combined = np.hstack((np.zeros(len(X_train)), np.ones(len(X_outTest))))  # 0 ��ʾѵ������1 ��ʾ���Լ�

# Use PCA to reduce data to 2D again
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)
# Split train and test after PCA again
X_train_pca = X_pca[y_combined == 0]
X_test_pca = X_pca[y_combined == 1]
# Calculate centroids for each class again
centroids = {
    label: np.mean(X_pca[y_combined == label], axis=0)
    for label in np.unique(y_combined)
}

# Calculate distance from each point to its class centroid again
distances = np.array([
    np.linalg.norm(X_pca[i] - centroids[y_combined[i]])
    for i in range(len(X_pca))
])

# Set outlier threshold at 95th percentile again
threshold = np.percentile(distances, 95)

# Identify outliers again
outliers = distances > threshold

# Remove outliers from X_outTest and y_outTest again
# X_outTest = X_outTest[~outliers[len(X_train):]]
# y_outTest = y_outTest[~outliers[len(X_train):]]
outlier_indices = np.where(outliers)[0]
outlier_points = X_pca[outlier_indices]
# Plot visualization again
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], label='Train', alpha=0.5, c='blue')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], label='Test', alpha=0.5, c='orange')

plt.scatter(outlier_points[:, 0], outlier_points[:, 1], label='Outliers', alpha=0.8, c='red', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization with Outliers')
plt.legend()
plt.savefig('./DIC/Data/output/pca_visualization1.png', dpi=300, bbox_inches='tight')  # dpi=300 ensures high resolution
print("Figure saved as pca_visualization1.png")

#endregion standardization
 # 6. List of models to be trained
print(f'-----------6. List of models to be trained')
#region  define models
models = {
    #"Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    #"SVM": SVC(probability=True),
    #"Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    #"Neural Networks": MLPClassifier(max_iter=200),
    #"Naive Bayes": GaussianNB(),
    #"AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier()
}
#endregion define models

 # 7. Parameter optimization

#region  parameter optimization
print(f'-----------7. Parameter optimization')
from sklearn.model_selection import GridSearchCV

param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    },
    "Random Forest": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "SVM": {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "Gradient Boosting": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "Neural Networks": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01]
    },
    "Naive Bayes": {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}


# Parameter optimization for each model --this part costs a lot of time, should skip when debugging.
#region   parameter optimization
# best_models = {}
# for name, model in models.items():
#     print(f"Optimizing {name}...")
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='accuracy')
#     grid_search.fit(X_train, y_train)
#     best_models[name] = grid_search.best_estimator_
#     print(f"Best parameters for {name}: {grid_search.best_params_}")
# models=best_models
#endregion parameter optimization

#endregion parameter optimization

 # 8. Model evaluation
#region  evaluate models
print(f'-----------8. Model evaluation')
feature_importances = pd.DataFrame(index=selected_features)

 # Create empty DataFrames for results
train_95CI = pd.DataFrame()
test_95CI = pd.DataFrame()
insidetest_95CI = pd.DataFrame()
outside_95CI = pd.DataFrame()

 # Initialize lists to store ROC curve data
roc_data = []
roc_data1 = []
roc_data2 = []
roc_data3 = []
# Store MCC results for each model
outMccResults = []
insideMccResults = []
trainMccResults = []
# Define scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    'log_loss': make_scorer(log_loss, greater_is_better=False, needs_proba=True),
    'mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False)
}

 # Create empty DataFrames for results
results = pd.DataFrame()
# Create empty DataFrames for bootstrap results
bootstrapResults = pd.DataFrame()
bootstrapResults1 = pd.DataFrame()
# Lists to store Precision-Recall curve data for each model
pr_curves = []
pr_curves1 = []
pr_curves2 = []
pr_curves3 = []
# Lists to store calibration curve data for each model
calibration_data = []
calibration_data1 = []
calibration_data2 = []
calibration_data3 = []

 

# ����ÿ��ģ�ͣ�ѵ����������
for name, model in models.items():
    model.fit(X_train, y_train)

    print(f"{datetime.datetime.now()}-- {name}--start 95CI...")    
    
    train_95CIitem = calculate_metrics(model, X_train, y_train, name)
    train_95CI = pd.concat([train_95CI,train_95CIitem]) 
    insidetest_95CIitem=calculate_metrics(model, X_insidetest, y_insidetest, name)
    insidetest_95CI = pd.concat([insidetest_95CI,insidetest_95CIitem])  
    outside_95CIitem=calculate_metrics(model, X_outTest, y_outTest, name)
    outside_95CI = pd.concat([outside_95CI,outside_95CIitem])
    print(f"{datetime.datetime.now()}-- {name}--end 95CI...")    
    
    
    print(f"{datetime.datetime.now()}-- {name}--start ������Ҫ��pfi...")    
    # ������Ҫ��pfi
    if hasattr(model, 'feature_importances_'):
        # �������ɭ��
        importances = model.feature_importances_
        feature_importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # �����߼��ع������SVM
        # importances = np.abs(model.coef_[0])  # ȡ����ֵ����Ϊϵ������Ϊ��
        importances = model.coef_[0]  # ȡ����ֵ����Ϊϵ������Ϊ��
        feature_importances[name] = model.coef_[0]

    print(f"{datetime.datetime.now()}-- {name}--end ������Ҫ��pfi...")    
    print(f"{datetime.datetime.now()}-- {name}--start outTest...") 
    # ����
    ################## start cross_validate
    print(f"{datetime.datetime.now()}-- {name}--start cross_validate()...")
    scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)  # 5�۽�����֤
    # ����ƽ���÷ֲ����浽DataFrame
    for key in scores:
        if key.startswith('test_'):
            for i, score in enumerate(scores[key]):
                results.loc[f'{name}_{i}', f'{key}'] = score
    ################## start bootstrap
    # ����һ���յ��ֵ�������ÿ��Bootstrap�ĵ÷�
    
    print(f"{datetime.datetime.now()}-- {name}--start bootstrap_scores()...")            
    bootstrap_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for i in range(100):  # ����100��Bootstrap
        # ����Bootstrap����
        X_resample, y_resample = resample(X_train, y_train)
        # ѵ��ģ�Ͳ�����÷�
        model.fit(X_resample, y_resample)
        y_pred = model.predict(X_train)
        bootstrap_scores['accuracy'].append(accuracy_score(y_train, y_pred))
        bootstrap_scores['precision'].append(precision_score(y_train, y_pred, average='weighted'))
        bootstrap_scores['recall'].append(recall_score(y_train, y_pred, average='weighted'))
        bootstrap_scores['f1'].append(f1_score(y_train, y_pred, average='weighted'))
    # ����ƽ���÷ֲ����浽DataFrame
    for metric in bootstrap_scores.keys():
        mean = np.mean(bootstrap_scores[metric])
        lower = np.percentile(bootstrap_scores[metric], 2.5)  # ����2.5�ٷ�λ�������������������
        
        upper = np.percentile(bootstrap_scores[metric], 97.5)  # ����97.5�ٷ�λ�������������������
        bootstrapResults.loc[name, f'{metric} Mean'] = mean
        bootstrapResults.loc[name, f'{metric} Lower 95% CI'] = lower
        bootstrapResults.loc[name, f'{metric} Upper 95% CI'] = upper
    print(f"{datetime.datetime.now()}-- {name}--end bootstrap_scores()...")       
    
    predictions = model.predict(X_outTest)
    predictions = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_outTest, predictions)
    print(f'{name} Accuracy: {accuracy * 100:.2f}%')
    # Get prediction probabilities
    y_score = model.predict_proba(X_outTest)[:, 1]
    
    thresholds, mcc_scores = calculate_mcc_curve(y_score, X_outTest, y_outTest)
    outMccResults.append((thresholds, mcc_scores, name))

    print(f"{datetime.datetime.now()}-- {name}--start outTest Compute ROC curve and ROC area...")     
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_outTest, y_score)
    roc_auc = auc(fpr, tpr)
    # ���� y_outTest �� y_score ����Ҫ���������
    # ���ȣ����ǽ�����ת��ΪDataFrame
    ddf = pd.DataFrame({
        'y_outTest': y_outTest,
        'y_score': y_score
    })

    # Ȼ�����ǿ���ʹ�� to_csv ������ DataFrame ����Ϊ CSV �ļ�
    ddf.to_csv('y_outTest-score.csv', index=False)
    # ����ROC��������
    roc_data.append((fpr, tpr, roc_auc, name))
    # ����ģ�͵�Precision-Recall����
    # ���� Precision-Recall ����
    precision, recall, thresholds = precision_recall_curve(y_outTest, y_score)

    # ���� F1 ����
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # �ҵ������ֵ
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"{datetime.datetime.now()}-- {name}--�����ֵ: {best_threshold}")
    threshold = best_threshold  # ������ֵΪ0.5
    predictions = (y_score >= threshold).astype(int)  # ��Ԥ�����ֵ������ֵ������Ϊ1��С�ڵ�����ֵ������Ϊ0
    
    # �������յľ�ȷ�ʡ��ٻ��ʺ� F1 ����
    final_precision = precision[np.argmax(f1_scores)]
    final_recall = recall[np.argmax(f1_scores)]
    final_f1 = f1_scores[np.argmax(f1_scores)]
    
    print(f"{datetime.datetime.now()}-- {name}--���վ�ȷ��: {final_precision}")
    print(f"{datetime.datetime.now()}-- {name}--�����ٻ���: {final_recall}")
    print(f"{datetime.datetime.now()}-- {name}--���� F1 ����: {final_f1}")

    precisionscore = precision_score(y_outTest, predictions)
    brier_score = brier_score_loss(y_outTest, y_score)
    # ��Precision-Recall���ߵ��������ӵ��б���
    pr_curves.append((precision, recall, name, precisionscore))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_outTest, y_score, n_bins=5)
    calibration_data.append((fraction_of_positives, mean_predicted_value, name,brier_score))

    ####################### insidetest
    print(f"{datetime.datetime.now()}-- {name}--start insidetest...") 
    predictions = model.predict(X_insidetest)
    
    accuracy = accuracy_score(y_insidetest, predictions)
    print(f'{name} insidetestData Accuracy: {accuracy * 100:.2f}%')
    # Get prediction probabilities
    y_score = model.predict_proba(X_insidetest)[:, 1]
    
    thresholds, mcc_scores = calculate_mcc_curve(y_score, X_insidetest, y_insidetest)
    insideMccResults.append((thresholds, mcc_scores, name))

    # Compute ROC curve and ROC area
    print(f"{datetime.datetime.now()}-- {name}--start insidetest Compute ROC curve and ROC area...") 
    fpr, tpr, _ = roc_curve(y_insidetest, y_score)
    roc_auc = auc(fpr, tpr)
    
    print(f"{datetime.datetime.now()}-- {name}--start ����ROC��������...")    
    # ����ROC��������
    roc_data1.append((fpr, tpr, roc_auc, name))
    # ����ģ�͵�Precision-Recall����
    threshold = 0.5  # ������ֵΪ0.5
    predictions = (y_score > threshold).astype(int)  # ��Ԥ�����ֵ������ֵ������Ϊ1��С�ڵ�����ֵ������Ϊ0
    precisionscore = precision_score(y_insidetest, predictions)
    precision, recall, _ = precision_recall_curve(y_insidetest, y_score)
    brier_score = brier_score_loss(y_insidetest, y_score)
    # ��Precision-Recall���ߵ��������ӵ��б���
    pr_curves1.append((precision, recall, name,precisionscore))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_insidetest, y_score, n_bins=5)
    calibration_data1.append((fraction_of_positives, mean_predicted_value, name,brier_score))
    
    #################### train
    print(f"{datetime.datetime.now()}-- {name}--start train...") 
    predictions = model.predict(X_train)
    accuracy = accuracy_score(y_train, predictions)
    print(f'{name} trainData Accuracy: {accuracy * 100:.2f}%')
    # Get prediction probabilities
    y_score = model.predict_proba(X_train)[:, 1]
        
    thresholds, mcc_scores = calculate_mcc_curve(y_score, X_train, y_train)
    trainMccResults.append((thresholds, mcc_scores, name))

    # Compute ROC curve and ROC area
    print(f"{datetime.datetime.now()}-- {name}--start train Compute ROC curve and ROC area...") 
    fpr, tpr, _ = roc_curve(y_train, y_score)
    roc_auc = auc(fpr, tpr)
    
    print(f"{datetime.datetime.now()}-- {name}--start ����ROC��������...")  
    # ����ROC��������
    roc_data3.append((fpr, tpr, roc_auc, name))
    # ����ģ�͵�Precision-Recall����
    threshold = 0.5  # ������ֵΪ0.5
    predictions = (y_score > threshold).astype(int)  # ��Ԥ�����ֵ������ֵ������Ϊ1��С�ڵ�����ֵ������Ϊ0
    precisionscore = precision_score(y_train, predictions)
    precision, recall, _ = precision_recall_curve(y_train, y_score)
    brier_score = brier_score_loss(y_train, y_score)
    # ��Precision-Recall���ߵ��������ӵ��б���
    pr_curves3.append((precision, recall, name,precisionscore))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_train, y_score, n_bins=5)
    calibration_data3.append((fraction_of_positives, mean_predicted_value, name,brier_score))
    
    # ��ȡ��ǰ���ں�ʱ��
    now = datetime.datetime.now()

    # �����ں�ʱ���ʽ��Ϊ�ַ���
    now_str = now.strftime('%Y%m%d_%H%M')    
    # ����SHAPֵ
    # ʹ��shap.sample�����ݽ��в���
    X_sample = shap.sample(X_train, nsamples=1)

    # ����ģ������ѡ�������
    if name in ["Random Forest", "Decision Tree", "Gradient Boosting", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
    elif name == "Neural Networks":
        try:
            explainer = shap.DeepExplainer(model, X_train)
        except Exception:
            try:
                shap.KernelExplainer(model.predict, X_sample)
            except Exception:
                continue
    elif name in ["Logistic Regression"]:
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict, X_sample)


    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    # ����SHAPֵ
    shap.summary_plot(shap_values, X_sample, max_display=15, feature_names=selected_features,show=False)
    plt.savefig(f'./DIC/Data/output/shap/{name}-shap_plot-{now_str}.png',dpi=700)
    plt.close()
    plt.clf()

# ��ȡ��ǰ���ں�ʱ��
now = datetime.datetime.now()

# �����ں�ʱ���ʽ��Ϊ�ַ���
now_str = now.strftime('%Y%m%d_%H%M%S')    
# ��������CSV�ļ�
results.to_csv(f'./DIC/Data/output/cross_validation_results_{now_str}.csv')
bootstrapResults.to_csv(f'./DIC/Data/output/bootstrap_results_{now_str}.csv')


train_95CI.to_csv(f'./DIC/Data/output/train_95CI_results_{now_str}.csv')
insidetest_95CI.to_csv(f'./DIC/Data/output/insidetest_95CI_results_{now_str}.csv')
test_95CI.to_csv(f'./DIC/Data/output/test_95CI_results_{now_str}.csv')
outside_95CI.to_csv(f'./DIC/Data/output/outside_95CI_results_{now_str}.csv')
   
# ���˳���Ҫ�Դ���0.05������
threshold = 0.0005
important_features = feature_importances[feature_importances > threshold].dropna(how='all')

# ���浽CSV�ļ�
important_features.to_csv(f'./DIC/Data/output/important_features_{now_str}.csv', encoding='UTF-8')    

# �����Ҫ������DIC�����������Ԫ�߼��ع�
logit_results = []
for feature in important_features[1:, 0]:
    if feature in allData.columns:
        X_logit = allData[[feature]].copy()
        X_logit = sm.add_constant(X_logit)
        y_logit = allData['DIC']
        # ȥ��ȱʧֵ
        mask = ~X_logit[feature].isnull() & ~y_logit.isnull()
        X_valid = X_logit[mask]
        y_valid = y_logit[mask]
        try:
            model = sm.Logit(y_valid, X_valid).fit(disp=0)
            coef = model.params[feature]
            se = model.bse[feature]
            or_val = np.exp(coef)
            ci_l = np.exp(coef - 1.96 * se)
            ci_u = np.exp(coef + 1.96 * se)
            p = model.pvalues[feature]
            logit_results.append({
                'Feature': feature,
                'OR': or_val,
                'CI_lower': ci_l,
                'CI_upper': ci_u,
                'P-value': p
            })
        except Exception as e:
            logit_results.append({
                'Feature': feature,
                'OR': np.nan,
                'CI_lower': np.nan,
                'CI_upper': np.nan,
                'P-value': np.nan
            })
logit_df = pd.DataFrame(logit_results)
logit_df.to_csv('./DIC/Data/output/logit_results.csv', index=False, encoding='utf-8')
print('��Ԫ�߼��ع����ѱ���Ϊ logit_results.csv')


# ����һ���µ�ͼ��
plt.figure()

# ����ÿ��ģ�͵�ROC�������ݣ�����ROC����
for fpr, tpr, roc_auc, name in roc_data:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Outside Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'./DIC/Data/output/Outside roc_curves_{now_str}.png')
#plt.show()

# ����һ���µ�ͼ��
plt.figure()

# ����ÿ��ģ�͵�ROC�������ݣ�����ROC����
for fpr, tpr, roc_auc, name in roc_data1:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Inside Test Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'./DIC/Data/output/Inside Test roc_curves_{now_str}.png')
# ����һ���µ�ͼ��
plt.figure()

# ����ÿ��ģ�͵�ROC�������ݣ�����ROC����
for fpr, tpr, roc_auc, name in roc_data3:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Train Data Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'./DIC/Data/output/Train Data roc_curves_{now_str}.png')
#plt.show()
# ����һ���µ�ͼ��
plt.figure()

# ����ÿ��ģ�͵�ROC�������ݣ�����ROC����
for fpr, tpr, roc_auc, name in roc_data2:
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{name} ROC curve (area = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Data Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show the plot
plt.savefig(f'./DIC/Data/output/Test Data roc_curves_{now_str}.png')
#plt.show()
plot_calibration_curves(calibration_data, 'Calibration curves', f'./DIC/Data/output/outside_data_calibration_curves_{now_str}.png')
plot_calibration_curves(calibration_data1, 'InsideTest Data Calibration curves', f'./DIC/Data/output/insidetest_data_calibration_curves_{now_str}.png')
plot_calibration_curves(calibration_data3, 'Train Data Calibration curves', f'./DIC/Data/output/train_data_calibration_curves_{now_str}.png')

plot_precision_recall_curves(pr_curves, 'Precision-Recall curve', f'./DIC/Data/output/outside_data_precision_recall_curve_{now_str}.png')
plot_precision_recall_curves(pr_curves1, 'Precision-Recall curve', f'./DIC/Data/output/insidetest_data_precision_recall_curve_{now_str}.png')
plot_precision_recall_curves(pr_curves3, 'Precision-Recall curve', f'./DIC/Data/output/train_data_precision_recall_curve_{now_str}.png')

plot_mcc_curves(outMccResults, f'./DIC/Data/output/outside_data_mcc_curve_{now_str}.png')
plot_mcc_curves(insideMccResults, f'./DIC/Data/output/inside_data_mcc_curve_{now_str}.png')
plot_mcc_curves(trainMccResults, f'./DIC/Data/output/train_data_mcc_curve_{now_str}.png')
#endregion evaluate models



