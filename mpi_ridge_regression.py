from mpi4py import MPI
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 初始化 MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 定义列名
column_names = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms', 
                'totalBedrooms', 'population', 'households', 
                'medianIncome', 'oceanProximity', 'medianHouseValue']

# 读取和预处理数据
def load_and_preprocess_data(file_path):
    # 读取数据，指定分隔符为 \t
    data = pd.read_csv(file_path, sep='\t', header=None, names=column_names)

    # 定义特征和目标
    X = data[['longitude', 'latitude', 'housingMedianAge', 'totalRooms', 
              'totalBedrooms', 'population', 'households', 'medianIncome', 
              'oceanProximity']]
    y = data['medianHouseValue']

    # 对'oceanProximity'做One-Hot编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['longitude', 'latitude', 'housingMedianAge', 
                                        'totalRooms', 'totalBedrooms', 'population', 
                                        'households', 'medianIncome']),
            ('cat', OneHotEncoder(), ['oceanProximity'])
        ]
    )

    # 数据预处理管道，返回 numpy 数组
    X = preprocessor.fit_transform(X)

    return X, y

# 计算局部核矩阵
def compute_local_kernel(X_chunk, X_block, gamma=0.01):  # 调整 gamma 值
    return rbf_kernel(X_chunk, X_block, gamma=gamma)

# 计算 RMSE
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 主进程负责加载和分配数据
if rank == 0:
    # 读取数据文件路径
    file_path = "D:\下载\housing.tsv"  # 替换为你的数据集路径
    X, y = load_and_preprocess_data(file_path)

    # 将数据划分为训练集和测试集，70% 训练集，30% 测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 将训练集划分为多个块，每个进程负责一块
    X_train_chunks = np.array_split(X_train, size, axis=0)
    y_train_chunks = np.array_split(y_train, size, axis=0)
    
    # 将测试集划分为多个块，每个进程负责一块
    X_test_chunks = np.array_split(X_test, size, axis=0)
    y_test_chunks = np.array_split(y_test, size, axis=0)

else:
    X_train_chunks = None
    y_train_chunks = None
    X_test_chunks = None
    y_test_chunks = None

# 分发训练集和测试集数据到所有进程
X_train_chunk = comm.scatter(X_train_chunks, root=0)
y_train_chunk = comm.scatter(y_train_chunks, root=0)
X_test_chunk = comm.scatter(X_test_chunks, root=0)
y_test_chunk = comm.scatter(y_test_chunks, root=0)

# 输出各进程的训练集和测试集大小
print(f"Rank {rank}: X_train_chunk shape = {X_train_chunk.shape}")
print(f"Rank {rank}: X_test_chunk shape = {X_test_chunk.shape}")

# 各进程收集所有其他进程的数据块
X_train_all = np.vstack(comm.allgather(X_train_chunk))

# 计算局部核矩阵
local_kernel_train = compute_local_kernel(X_train_chunk, X_train_all)
print(f"Rank {rank}: Computed local kernel matrix of shape {local_kernel_train.shape}")

# 收集每个进程的局部核矩阵，并在主进程拼接成全局核矩阵
kernel_train_all = comm.gather(local_kernel_train, root=0)

# 主进程拼接局部核矩阵并进行核岭回归训练
if rank == 0:
    global_kernel_train = np.vstack(kernel_train_all)

    # 调整正则化项 alpha
    alpha = 10.0  # 提高正则化强度
    global_kernel_train += alpha * np.eye(global_kernel_train.shape[0])

    # 解 alpha_ (根据 K * alpha_ = y_train)
    alpha_ = np.linalg.solve(global_kernel_train, y_train)

    print(f"Shape of global kernel train: {global_kernel_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of alpha_: {alpha_.shape}")
else:
    alpha_ = None

# 广播 alpha_ 到所有进程
alpha_ = comm.bcast(alpha_, root=0)

# 每个进程计算其负责的测试集核矩阵
K_test_chunk = compute_local_kernel(X_test_chunk, X_train_all)

# 每个进程计算自己负责的测试集预测结果
y_pred_test_chunk = K_test_chunk.dot(alpha_)

# 收集每个进程的测试集预测结果
y_pred_test_all = comm.gather(y_pred_test_chunk, root=0)
y_test_all = comm.gather(y_test_chunk, root=0)

# 主进程计算 RMSE
if rank == 0:
    # 将所有进程的预测结果和真实结果进行拼接
    y_pred_test = np.hstack(y_pred_test_all)
    y_test_concat = np.hstack(y_test_all)

    # 确保拼接后的预测结果与 y_test 的长度一致
    assert y_pred_test.shape[0] == y_test_concat.shape[0], f"Length mismatch: {y_pred_test.shape[0]} vs {y_test_concat.shape[0]}"

    # 计算训练集的预测值
    y_pred_train = global_kernel_train.dot(alpha_)

    # 计算训练集和测试集的 RMSE
    train_rmse = compute_rmse(y_train, y_pred_train)
    test_rmse = compute_rmse(y_test_concat, y_pred_test)

    # 输出结果
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
