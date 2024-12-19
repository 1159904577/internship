# ipynb------>py
# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import plot_importance
from scipy.stats import skew, kurtosis
import joblib
from xgboost import XGBClassifier


# 读取数据文件
FILE = 'C:\\Users\\13719\\Documents\\internship\\DGA代码\\artifacts\\binary\\dga_binary_test.csv'
df = pd.read_csv(FILE)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱数据顺序
df.head()  # 显示数据的前几行

# 绘制Botnet_Family列的分布图（排除'alexa'类别）
sns.countplot(df[df['Botnet_Family']!='alexa']['Botnet_Family'])

# 查看目标标签（Target）各个类别的数量
df['Target'].value_counts()

# 只保留域名和目标标签列
df = df[['Domain', 'Target']]
df.head()

# 定义函数：提取域名的特征
def count_features(domain):
    # 计算域名的长度
    L = len(domain)

    # 计算域名中辅音字符的比例
    consonant_count = sum(1 for char in domain if char in 'bcdfghjklmnpqrstvwxyz')
    Rc = consonant_count / len(domain) if len(domain) > 0 else 0

    # 计算域名中字母字符的比例
    letter_count = sum(1 for char in domain if char.isalpha())
    Rl = letter_count / len(domain) if len(domain) > 0 else 0

    # 计算域名中数字字符的比例
    number_count = sum(1 for char in domain if char.isdigit())
    Rn = number_count / len(domain) if len(domain) > 0 else 0

    # 计算域名中元音字符的比例
    vowel_count = sum(1 for char in domain if char in 'aeiou')
    Rv = vowel_count / len(domain) if len(domain) > 0 else 0

    # 计算域名中符号字符的比例
    symbolic_count = sum(1 for char in domain if not char.isalnum())
    Rs = symbolic_count / len(domain) if len(domain) > 0 else 0

    # 返回所有特征
    return L, Rc, Rv, Rn, Rl, Rs

# 定义函数：计算每个域名的多个特征
def calculate_features(df):
    features = []
    for domain in df['Domain']:
        parts = domain.split('.')  # 按照'.'分割域名
        subdomain = '.'.join(parts[:-2]) if len(parts) >= 3 else ''  # 获取子域名
        sld = parts[-2]  # 获取二级域名
        tld = parts[-1]  # 获取顶级域名

        # 计算域名级别数量
        N = 3 if subdomain else 2

        # 计算最长连续辅音序列的长度
        consonants = re.findall(r'[^aeiou\d\s\W]+', domain)
        LCc = max(len(consonant) for consonant in consonants) if consonants else 0

        # 计算最长连续数字序列的长度
        numbers = re.findall(r'\d+', domain)
        LCn = max(len(number) for number in numbers) if numbers else 0

        # 计算最长连续元音序列的长度
        vowels = re.findall(r'[aeiou]+', domain)
        LCv = max(len(vowel) for vowel in vowels) if vowels else 0

        # 获取每个域名级别的特征
        L_tld, Rc_tld, Rv_tld, Rn_tld, Rl_tld, Rs_tld = count_features(tld)
        L_sld, Rc_sld, Rv_sld, Rn_sld, Rl_sld, Rs_sld = count_features(sld)
        L_sub, Rc_sub, Rv_sub, Rn_sub, Rl_sub, Rs_sub = count_features(subdomain) if subdomain else (0, 0, 0, 0, 0, 0)

        # 将特征添加到列表中
        features.append([N, LCc, LCv, LCn, L_tld, Rc_tld, Rv_tld, Rn_tld, Rl_tld, Rs_tld,
                         L_sld, Rc_sld, Rv_sld, Rn_sld, Rl_sld, Rs_sld, L_sub, Rc_sub, Rv_sub, Rn_sub, Rl_sub, Rs_sub])

    # 定义特征列名
    feature_columns = ['N', 'LCc', 'LCv', 'LCn', 'L_tld', 'Rc_tld', 'Rv_tld', 'Rn_tld', 'Rl_tld', 'Rs_tld',
                       'L_sld', 'Rc_sld', 'Rv_sld', 'Rn_sld', 'Rl_sld', 'Rs_sld', 'L_sub', 'Rc_sub', 'Rv_sub',
                       'Rn_sub', 'Rl_sub', 'Rs_sub']

    # 将特征列表转换为DataFrame
    feature_df = pd.DataFrame(features, columns=feature_columns)

    # 返回拼接后的DataFrame
    return pd.concat([df, feature_df], axis=1)

# 计算每个域名的特征
df_custom_features = calculate_features(df)
df_custom_features.head()

# 创建n-gram特征提取器
unigrams = TfidfVectorizer(analyzer='char', ngram_range=(1, 1))  # 单字符特征
bigrams = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))  # 双字符特征
trigrams = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))  # 三字符特征

# 提取单字符、双字符和三字符特征
unigrams_matrix = unigrams.fit_transform(df['Domain'])
bigrams_matrix = bigrams.fit_transform(df['Domain'])
trigrams_matrix = trigrams.fit_transform(df['Domain'])

# 定义函数：提取每个样本的n-gram特征
def ngrams_features_per_sample(matrix, prefix):
    ngram_frequencies = matrix  # 保持稀疏矩阵格式，不转换为密集矩阵
    features_list = []
    
    # 遍历每个样本，计算其n-gram特征
    for sample_frequencies in ngram_frequencies:
        features = {}
        if sample_frequencies.nnz > 0:  # 如果该样本有n-gram特征
            # 计算各类统计特征
            features[f'{prefix}-DIST'] = sample_frequencies.nnz  # nnz是稀疏矩阵中非零元素的数量
            features[f'{prefix}-MEAN'] = sample_frequencies.mean()
            features[f'{prefix}-QMEAN'] = np.sqrt(np.mean(sample_frequencies.data**2))
            features[f'{prefix}-SUMSQ'] = np.sum(sample_frequencies.data**2)
            features[f'{prefix}-VAR'] = np.var(sample_frequencies.data)
            features[f'{prefix}-PVAR'] = np.var(sample_frequencies.data, ddof=0)
            features[f'{prefix}-STD'] = np.std(sample_frequencies.data)
            features[f'{prefix}-PSTD'] = np.std(sample_frequencies.data, ddof=0)
            features[f'{prefix}-SKE'] = skew(sample_frequencies.data)
            features[f'{prefix}-KUR'] = kurtosis(sample_frequencies.data)
        else:  # 如果该样本没有n-gram特征
            features[f'{prefix}-DIST'] = 0
            features[f'{prefix}-MEAN'] = 0
            features[f'{prefix}-QMEAN'] = 0
            features[f'{prefix}-SUMSQ'] = 0
            features[f'{prefix}-VAR'] = 0
            features[f'{prefix}-PVAR'] = 0
            features[f'{prefix}-STD'] = 0
            features[f'{prefix}-PSTD'] = 0
            features[f'{prefix}-SKE'] = 0
            features[f'{prefix}-KUR'] = 0

        features_list.append(features)

    return pd.DataFrame(features_list)


# 提取单字符、双字符、三字符的n-gram特征
unigrams_features_df = ngrams_features_per_sample(unigrams_matrix, prefix='UNI')
bigrams_features_df = ngrams_features_per_sample(bigrams_matrix, prefix='BI')
trigrams_features_df = ngrams_features_per_sample(trigrams_matrix, prefix='TRI')

# 拼接n-gram特征
df_ngrams_features = pd.concat([unigrams_features_df, bigrams_features_df, trigrams_features_df], axis=1)

# 拼接所有特征
df_final = pd.concat([df_custom_features, df_ngrams_features], axis=1)

df_final.head()

# 准备训练数据
X = df_final.drop(['Domain', 'Target'], axis=1)
y = df_final['Target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义要训练的模型
models = {
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'k-Nearest Neighbours': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# 设置交叉验证次数
k = 5
results = {'Accuracy': {}, 'F1 Score': {}}
roc_curves = {}

# 对每个模型进行训练、评估和ROC曲线绘制
for model_name, model in models.items():
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)  # 设置交叉验证
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')  # 计算准确率
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')  # 计算F1分数
    
    results['Accuracy'][model_name] = accuracy_scores.mean()  # 记录平均准确率
    results['F1 Score'][model_name] = f1_scores.mean()  # 记录平均F1分数

    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 预测测试集
    
    # 计算ROC曲线和AUC值
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    roc_curves[model_name] = (fpr, tpr, roc_auc)

results, roc_curves

# 再次训练和绘制ROC曲线
for model_name, model in models.items():
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)  # 设置交叉验证
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    
    results['Accuracy'][model_name] = accuracy_scores.mean()  # 记录准确率
    results['F1 Score'][model_name] = f1_scores.mean()  # 记录F1分数

    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 预测
    y_score = model.predict_proba(X_test)[:, 1]  # 获取预测的概率值（用于绘制ROC曲线）
    
    # 计算并绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    roc_curves[model_name] = (fpr, tpr, roc_auc)
    
    print(f"{model_name} Accuracy: {accuracy_scores.mean():.4f} | F1 Score: {f1_scores.mean():.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # 绘制ROC曲线
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')

# 绘制随机模型的对比线
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of Models')
plt.legend(loc='best')
plt.show()

# 生成pkl模型
# joblib.dump(unigrams, 'unigram_vectorizer.pkl')
# joblib.dump(bigrams, 'bigram_vectorizer.pkl')
# joblib.dump(trigrams, 'trigram_vectorizer.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(xgb_model, 'binary_classification_model.pkl')