#访问UCI机器学习数据集为实验进行数据支撑。该数据集包含随访期间收集的29 9名心力衰竭患者的病历，其中每个患者档案有 13 个临床特征。目标为死亡事件：患者是否在随访期间死亡。
from ucimlrepo import fetch_ucirepo
import pandas as pd
# fetch dataset
heart_failure_clinical_records = fetch_ucirepo(id=519)#该数据集的索引（ID）为 519
# 将特征数据转换为 pandas DataFrame
features_df = pd.DataFrame(heart_failure_clinical_records.data.features)
# 显示前5行
print(features_df.head())

X = heart_failure_clinical_records.data.features
y = heart_failure_clinical_records.data.targets

# metadata,查看数据集的元数据
print(heart_failure_clinical_records.metadata)
# variable information
print(heart_failure_clinical_records.variables)
# 获取样本数量
num_samples = len(heart_failure_clinical_records.data.targets)
print("数据集中共有{}个样本。".format(num_samples))

from sklearn import datasets #Scikit-Learn涵盖了几乎所有主流的机器学习算法，包括分类、回归、聚类、降维等任务
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_curve, auc
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 预测并计算性能指标（如AUC）
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc_nb = auc(fpr,tpr)

#显示混淆矩阵热力图
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns #数据可视化库，提供了多种美观的调色板，可以自定义图表的颜色，多种图表如条形图、热力图
import matplotlib.pyplot as plt  #生成高质量的图表和图形
plt.rcParams['font.sans-serif'] = 'SimHei'#对图表进行参数配置，设置字体为simhei字体
plt.rcParams['axes.unicode_minus'] = False#确保负号能正确显示
# 输出混淆矩阵
confusion_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (6, 6), dpi = 300) # 创建一个大小为6x6英寸，分辨率为300 DPI的图形
sns.heatmap(confusion_matrix, annot = True, annot_kws = {'size':15}, fmt = 'd', cmap = 'YlGnBu_r')
# 绘制热力图 annot = True热力图上显示数值，annot_kws设置单元格数值标签的其他属性,fmt指定单元格数据显示格式，'d' 表示整数格式，如果你的数据是浮点数，可以使用 '.2f' 来显示两位小数，cmap用于热力图填色
plt.title('混淆矩阵热力图')
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.show()


# 绘制ROC曲线
plt.figure(2)
lw = 2  #线宽为2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_nb)#area的值为roc_auc_nb取2位浮点数
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')#绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()

# 输出模型报告， 查看评价指标
print(classification_report(y_test, y_pred))