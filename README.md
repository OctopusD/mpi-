# mpi_ridge_regression
## Overview
This repository provides a method for solving Kernel Ridge Regression using MPI parallel computing and the Conjugate Gradient method. Kernel Ridge Regression is a regularized regression method that extends linear models using kernel functions. This implementation is optimized for large-scale datasets, leveraging parallel processing and efficient algorithms to enhance computational performance.

## Method Description
### 1. MPI Parallelization Initialization
The program uses the mpi4py library to implement parallelization. Each process in MPI is uniquely identified by a rank, and multiple processes collaborate to divide tasks and process data.

 `comm = MPI.COMM_WORLD `: Retrieves the global communicator containing all processes. 
Typically, the master process is rank 0, responsible for task distribution, while other processes handle their respective data chunks.
### 2. Data Preprocessing
Data is read using pandas, with features including geographical information (longitude, latitude), housing attributes (`totalRooms`, `medianIncome`), and categorical features (`oceanProximity`).

Standardization and One-Hot Encoding are applied to preprocess features. Standardization ensures that features with different scales are on the same magnitude, while One-Hot Encoding handles categorical data.

python

` preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['longitude', 'latitude', ...]),
        ('cat', OneHotEncoder(), ['oceanProximity'])
    ]
)
`
### 3. Data Partitioning and Distribution
The dataset is split into training (70%) and testing (30%) sets using `train_test_split`.

The training and testing sets are further split into chunks using `np.array_split`, with each MPI process responsible for processing its own data chunk. This parallel processing reduces the computational load for each process.

python

`X_train_chunks = np.array_split(X_train, size, axis=0)`
### 4. Kernel Matrix Computation
The Radial Basis Function Kernel (RBF Kernel) is used to compute the kernel matrix for the input data. Each process computes the local kernel matrix for its assigned data chunk, reducing the overall computation.

python

`def compute_local_kernel(X_chunk, X_block, gamma=0.01):
    return rbf_kernel(X_chunk, X_block, gamma=gamma)`
5. Solving Kernel Ridge Regression with Conjugate Gradient Method
The Conjugate Gradient method (CG) is employed to solve for the alpha_ coefficients in Kernel Ridge Regression. CG is an efficient iterative algorithm particularly suited for large, sparse linear systems. It significantly reduces computational complexity compared to directly solving linear systems (e.g., matrix inversion).

python

`def conjugate_gradient_solve(K, y, alpha, tol=1e-6, max_iter=1000):
    K_reg = K + alpha * np.eye(K.shape[0])
    beta, info = cg(K_reg, y, tol=tol, maxiter=max_iter)
    return beta, info
K_reg = K + alpha * I`: The regularization term alpha is added to the kernel matrix K to prevent overfitting.

The cg() function returns the coefficients beta, which are equivalent to the regression coefficients alpha_, and info provides information on whether the CG method has converged.

### 6. Training and Prediction
Training Phase: The master process computes the global kernel matrix for the training data and solves for the regression coefficients alpha_ using the Conjugate Gradient method.
Prediction Phase: Each process computes predictions for its own test data chunk and sends the results back to the master process. The master process aggregates all predictions and compares them with the true values.
### 7. Performance Evaluation
The model performance is evaluated using Root Mean Squared Error (RMSE) on both the training and test sets.

python

`def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))`
Training RMSE: Measures how well the model fits the training data.

Test RMSE: Reflects the model’s generalization ability. A high test RMSE compared to the training RMSE may indicate overfitting.

Test Results
Training Error (Training RMSE):

The RMSE on the training set is very low (potentially close to 0), indicating that the model fits the training data extremely well. This is expected since the Conjugate Gradient method finds the exact optimal solution.
Test Error (Test RMSE):

The test RMSE provides a better measure of the model’s ability to generalize. A significant increase in test RMSE relative to training RMSE may suggest overfitting. Adjusting the regularization parameter alpha can help reduce overfitting.
Convergence of Conjugate Gradient Method:

The info variable indicates whether the Conjugate Gradient method converged. If info == 0, the algorithm successfully converged. If not, adjusting the number of iterations (max_iter) or the convergence tolerance (tol) may be necessary.
Conclusion
The combination of the Conjugate Gradient method and parallel computing provides an efficient approach to solving Kernel Ridge Regression on large datasets. While the training error is often low, it is essential to monitor the model's generalization ability by analyzing the test error. Adjusting the regularization parameter helps mitigate overfitting. Additionally, MPI parallelization significantly improves kernel matrix computation efficiency, enabling the method to scale to larger datasets.

Requirements
mpi4py
scikit-learn
pandas
numpy
How to Run
Install the required dependencies.

Run the program using an MPI launcher, e.g.:

bash

`mpirun -n <num_processes> python mpi_ridge_regression.py`
