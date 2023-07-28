# Multi-linear-regression-using-Artificial-Neural-Network
Ann multi linear regression model on housing data set to predict house prices

# Data Preparation:

X_data: This variable holds the independent features (input variables) of the dataset. It is obtained by dropping the "Price" column from the DataFrame df.
Y_data: This variable represents the dependent feature or target variable, which is the "Price" column from the DataFrame df.

# Data Splitting and Scaling:

The code splits the data into training and test sets. 80% of the data is used for training, and the remaining 20% is used for testing.
x_train: This variable contains the independent features of the training set (80% of X_data).
y_train: This variable contains the corresponding target values for the training set, which are extracted from Y_data using the indices of x_train.
x_test: This variable contains the independent features of the test set (remaining 20% of X_data).
y_test: This variable contains the corresponding target values for the test set, which are extracted from Y_data using the indices of the test set.

# Data Preprocessing for Neural Network:

The code converts the data from DataFrames to NumPy arrays and reshapes them to fit the expected input shape of the neural network.

# Building the Neural Network Model:

The neural network is constructed using the Sequential API from TensorFlow/Keras.
It consists of three dense layers with different activation functions:
The first hidden layer has 20 neurons with a ReLU activation function and an input shape of (1, 5).
The second hidden layer has 10 neurons with a ReLU activation function.
The output layer has 1 neuron with a ReLU activation function, which is appropriate for regression tasks.

# Model Compilation:

The model is compiled with the Mean Squared Error (MSE) loss function, which is commonly used for regression problems.
The Adam optimizer is used for training the model.

# Model Training:

The model is trained using the fit method with x_train and y_train (which might be a mistake) as inputs for a certain number of epochs and a batch size of 32.

# Model Evaluation:

The model is evaluated on the test set using the evaluate method, which computes the loss (MSE) on the test data.

# Making Predictions:

The model is used to predict the target values for the test set, and the predictions are stored in y_pred.

# Visualization:

The code creates a line plot with two lines representing the actual and predicted target values from the test set. The plot is intended to visualize the performance of the model.
