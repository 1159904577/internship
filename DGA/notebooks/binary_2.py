# 分批处理
# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from scipy.stats import skew, kurtosis
import joblib
import gc  # 用于垃圾回收

# 读取数据文件
FILE = 'C:\\Users\\13719\\Documents\\internship\\DGA代码\\artifacts\\binary\\dga_binary_test.csv'
df = pd.read_csv(FILE)

# 打乱数据顺序
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 仅保留域名和目标标签
df = df[['Domain', 'Target']]

# 定义函数：提取域名的特征
def count_features(domain):
    L = len(domain)
    consonant_count = sum(1 for char in domain if char in 'bcdfghjklmnpqrstvwxyz')
    Rc = consonant_count / L if L > 0 else 0
    letter_count = sum(1 for char in domain if char.isalpha())
    Rl = letter_count / L if L > 0 else 0
    number_count = sum(1 for char in domain if char.isdigit())
    Rn = number_count / L if L > 0 else 0
    vowel_count = sum(1 for char in domain if char in 'aeiou')
    Rv = vowel_count / L if L > 0 else 0
    symbolic_count = sum(1 for char in domain if not char.isalnum())
    Rs = symbolic_count / L if L > 0 else 0
    return L, Rc, Rv, Rn, Rl, Rs

# 定义函数：计算每个域名的多个特征
def calculate_features(df):
    features = []
    for domain in df['Domain']:
        parts = domain.split('.')
        subdomain = '.'.join(parts[:-2]) if len(parts) >= 3 else ''
        sld = parts[-2]
        tld = parts[-1]
        N = 3 if subdomain else 2
        consonants = re.findall(r'[^aeiou\d\s\W]+', domain)
        LCc = max(len(consonant) for consonant in consonants) if consonants else 0
        numbers = re.findall(r'\d+', domain)
        LCn = max(len(number) for number in numbers) if numbers else 0
        vowels = re.findall(r'[aeiou]+', domain)
        LCv = max(len(vowel) for vowel in vowels) if vowels else 0
        L_tld, Rc_tld, Rv_tld, Rn_tld, Rl_tld, Rs_tld = count_features(tld)
        L_sld, Rc_sld, Rv_sld, Rn_sld, Rl_sld, Rs_sld = count_features(sld)
        L_sub, Rc_sub, Rv_sub, Rn_sub, Rl_sub, Rs_sub = count_features(subdomain) if subdomain else (0, 0, 0, 0, 0, 0)
        features.append([N, LCc, LCv, LCn, L_tld, Rc_tld, Rv_tld, Rn_tld, Rl_tld, Rs_tld,
                         L_sld, Rc_sld, Rv_sld, Rn_sld, Rl_sld, Rs_sld, L_sub, Rc_sub, Rv_sub, Rn_sub, Rl_sub, Rs_sub])
    feature_columns = ['N', 'LCc', 'LCv', 'LCn', 'L_tld', 'Rc_tld', 'Rv_tld', 'Rn_tld', 'Rl_tld', 'Rs_tld',
                       'L_sld', 'Rc_sld', 'Rv_sld', 'Rn_sld', 'Rl_sld', 'Rs_sld',
                       'L_sub', 'Rc_sub', 'Rv_sub', 'Rn_sub', 'Rl_sub', 'Rs_sub']
    feature_df = pd.DataFrame(features, columns=feature_columns)
    return pd.concat([df.reset_index(drop=True), feature_df], axis=1)

# 计算每个域名的自定义特征
print("计算自定义特征...")
df_custom_features = calculate_features(df)
print("自定义特征计算完成。")

# 创建n-gram特征提取器，并限制最大特征数量
print("创建n-gram向量器...")
unigrams = TfidfVectorizer(analyzer='char', ngram_range=(1, 1), max_features=1000)
bigrams = TfidfVectorizer(analyzer='char', ngram_range=(2, 2), max_features=1000)
trigrams = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=1000)

# 仅在所有数据上拟合向量器
print("拟合n-gram向量器...")
unigrams.fit(df['Domain'])
bigrams.fit(df['Domain'])
trigrams.fit(df['Domain'])
print("n-gram向量器拟合完成。")

# 定义n-gram统计特征提取函数
def ngrams_features_per_sample(matrix, prefix):
    ngram_frequencies = matrix
    features_list = []
    for sample_frequencies in ngram_frequencies:
        features = {}
        if sample_frequencies.nnz > 0:
            data = sample_frequencies.data
            features[f'{prefix}-DIST'] = sample_frequencies.nnz
            features[f'{prefix}-MEAN'] = data.mean()
            features[f'{prefix}-QMEAN'] = np.sqrt(np.mean(data ** 2))
            features[f'{prefix}-SUMSQ'] = np.sum(data ** 2)
            features[f'{prefix}-VAR'] = np.var(data)
            features[f'{prefix}-PVAR'] = np.var(data, ddof=0)
            features[f'{prefix}-STD'] = np.std(data)
            features[f'{prefix}-PSTD'] = np.std(data, ddof=0)
            features[f'{prefix}-SKE'] = skew(data) if len(data) > 2 else 0
            features[f'{prefix}-KUR'] = kurtosis(data) if len(data) > 3 else 0
        else:
            for metric in ['DIST', 'MEAN', 'QMEAN', 'SUMSQ', 'VAR', 'PVAR', 'STD', 'PSTD', 'SKE', 'KUR']:
                features[f'{prefix}-{metric}'] = 0
        features_list.append(features)
    return pd.DataFrame(features_list)

# 定义批处理函数
def process_batch(domains, unigrams, bigrams, trigrams):
    unigrams_matrix = unigrams.transform(domains)
    bigrams_matrix = bigrams.transform(domains)
    trigrams_matrix = trigrams.transform(domains)
    
    unigrams_features = ngrams_features_per_sample(unigrams_matrix, prefix='UNI')
    bigrams_features = ngrams_features_per_sample(bigrams_matrix, prefix='BI')
    trigrams_features = ngrams_features_per_sample(trigrams_matrix, prefix='TRI')
    
    ngram_features = pd.concat([unigrams_features, bigrams_features, trigrams_features], axis=1)
    return ngram_features

# 分批处理n-gram特征
print("开始分批提取n-gram特征...")
batch_size = 10000
num_batches = int(np.ceil(len(df) / batch_size))
df_ngrams_features = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(df))
    batch_domains = df['Domain'].iloc[start_idx:end_idx]
    batch_features = process_batch(batch_domains, unigrams, bigrams, trigrams)
    df_ngrams_features.append(batch_features)
    print(f"已处理批次 {i+1}/{num_batches}")
    
    # 释放内存
    del batch_domains, batch_features
    gc.collect()

# 合并所有批次的n-gram特征
df_ngrams_features = pd.concat(df_ngrams_features, axis=0).reset_index(drop=True)
print("n-gram特征提取完成。")

# 释放内存
del df
gc.collect()

# 拼接所有特征
print("拼接自定义特征和n-gram特征...")
df_final = pd.concat([df_custom_features, df_ngrams_features], axis=1)
print("特征拼接完成。")

# 准备训练数据
X = df_final.drop(['Domain', 'Target'], axis=1)
y = df_final['Target']

# 释放内存
del df_custom_features, df_ngrams_features
gc.collect()

# 划分训练集和测试集
print("划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("数据集划分完成。")

# 释放内存
del X, y
gc.collect()

# 数据标准化
print("标准化数据...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("数据标准化完成。")

# 定义要训练的模型
models = {
    'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'k-Nearest Neighbours': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# 设置交叉验证次数
k = 5
results = {'Accuracy': {}, 'F1 Score': {}}
roc_curves = {}

# 对每个模型进行训练、评估和ROC曲线绘制
print("开始训练和评估模型...")
for model_name, model in models.items():
    print(f"训练模型: {model_name}")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    
    results['Accuracy'][model_name] = accuracy_scores.mean()
    results['F1 Score'][model_name] = f1_scores.mean()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 计算ROC曲线和AUC值
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    roc_curves[model_name] = (fpr, tpr, roc_auc)
    
    print(f"{model_name} - Accuracy: {accuracy_scores.mean():.4f} | F1 Score: {f1_scores.mean():.4f} | ROC AUC: {roc_auc:.4f}")

print("模型训练和评估完成。")

# 绘制ROC曲线
print("绘制ROC曲线...")
plt.figure(figsize=(10, 8))

for model_name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')

# 绘制随机模型的对比线
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of Models')
plt.legend(loc='best')
plt.grid(True)
plt.show()
print("ROC曲线绘制完成。")

# 打印总结结果
print("\n模型评估结果:")
results_df = pd.DataFrame(results).T
print(results_df)

# 保存模型和向量器
# print("保存模型和向量器...")
# joblib.dump(unigrams, 'unigram_vectorizer.pkl')
# joblib.dump(bigrams, 'bigram_vectorizer.pkl')
# joblib.dump(trigrams, 'trigram_vectorizer.pkl')
# joblib.dump(scaler, 'scaler.pkl')

# # 仅保存XGBoost模型作为示例，您可以根据需要保存其他模型
# joblib.dump(models['XGBoost'], 'xgb_model.pkl')
# print("模型和向量器保存完成。")
