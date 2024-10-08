import numpy as np
from tqdm import tqdm

class SoftmaxRegression:
    def __init__(self, num_classes, num_iterations=100, random_seed=None, lr_steplizer=None):

        self.num_classes = num_classes
        # learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.weights = None
        self.lr_steplizer = lr_steplizer


    def fit(self, X_train, y_train, X_val=None, y_val=None):

        # Add bias term to the training data
        X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        
        # Initialize weights with random values
        np.random.seed(self.random_seed)
        self.weights = np.random.normal(size=(X_train_bias.shape[1], self.num_classes))

        # Lists to store training and validation losses during training
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for iteration in tqdm(range(self.num_iterations)):
            # Calculate logits and softmax probabilities
            logits = np.dot(X_train_bias, self.weights)
            exp_logits = np.exp(logits)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            self.lr_steplizer.step()

            learning_rate = self.lr_steplizer.get_lr()
            # Compute the cross-entropy loss
            loss = -np.mean(y_train * np.log(softmax_probs))
            
            # Compute the gradient and update weights
            gradient = np.dot(X_train_bias.T, softmax_probs - y_train) / X_train_bias.shape[0]
            self.weights -= learning_rate * gradient

            # Calculate training accuracy
            train_pred = np.argmax(softmax_probs, axis=1)
            train_accuracy = np.mean(train_pred == np.argmax(y_train, axis=1))
            
            train_accuracies.append(train_accuracy)
            train_losses.append(loss)


            if X_val is not None and y_val is not None:
                # Calculate validation loss
                X_val_bias = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
                logits_val = np.dot(X_val_bias, self.weights)
                exp_logits_val = np.exp(logits_val)
                softmax_probs_val = exp_logits_val / np.sum(exp_logits_val, axis=1, keepdims=True)
                val_loss = -np.mean(y_val * np.log(softmax_probs_val))
                
                # Calculate validation accuracy
                val_pred = np.argmax(softmax_probs_val, axis=1)
                val_accuracy = np.mean(val_pred == np.argmax(y_val, axis=1))
                
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
            

        return train_losses, val_losses, train_accuracies, val_accuracies

    def continue_fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Multinomial Logistic Regression model.

        Parameters:
        - X_train: Training feature data.
        - y_train: Training labels.
        - X_val: Validation feature data (optional).
        - y_val: Validation labels (optional).

        Returns:
        - train_losses: List of training losses during iterations.
        - train_accuracies: List of training accuracies during iterations.
        - val_losses: List of validation losses during iterations (if validation data provided).
        - val_accuracies: List of validation accuracies during iterations (if validation data provided).
        """
        # # Add bias term to the training data
        X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        
        # # Initialize weights with random values
        # np.random.seed(self.random_seed)
        # self.weights = np.random.randn(X_train_bias.shape[1], self.num_classes)

        # # Lists to store training and validation losses during training
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for iteration in tqdm(range(self.num_iterations)):
            # Calculate logits and softmax probabilities
            logits = np.dot(X_train_bias, self.weights)
            exp_logits = np.exp(logits)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            self.lr_steplizer.step()

            learning_rate = self.lr_steplizer.get_lr()
            # Compute the cross-entropy loss
            loss = -np.mean(y_train * np.log(softmax_probs))
            
            # Compute the gradient and update weights
            gradient = np.dot(X_train_bias.T, softmax_probs - y_train) / X_train_bias.shape[0]
            self.weights -= learning_rate * gradient

            # Calculate training accuracy
            train_pred = np.argmax(softmax_probs, axis=1)
            train_accuracy = np.mean(train_pred == np.argmax(y_train, axis=1))
            
            train_accuracies.append(train_accuracy)
            train_losses.append(loss)


            if X_val is not None and y_val is not None:
                # Calculate validation loss
                X_val_bias = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
                logits_val = np.dot(X_val_bias, self.weights)
                exp_logits_val = np.exp(logits_val)
                softmax_probs_val = exp_logits_val / np.sum(exp_logits_val, axis=1, keepdims=True)
                val_loss = -np.mean(y_val * np.log(softmax_probs_val))
                
                # Calculate validation accuracy
                val_pred = np.argmax(softmax_probs_val, axis=1)
                val_accuracy = np.mean(val_pred == np.argmax(y_val, axis=1))
                
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
            

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Feature data for prediction.

        Returns:
        - predicted_class: Predicted class labels.
        """
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        logits = np.dot(X_bias, self.weights)
        predicted_class = np.argmax(logits, axis=1)
        return predicted_class

# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import OneHotEncoder
 
 
# class SoftmaxRegression:
#     def __init__(self, num_classes, learning_rate=0.01, num_iterations=100, random_seed=None):
#         """
#         Initialize the Multinomial Logistic Regression model.
 
#         Parameters:
#         - num_classes: The number of classes for classification.
#         - learning_rate: The learning rate for gradient descent (default is 0.01).
#         - num_iterations: The number of training iterations (default is 100).
#         - random_seed: int, random seed for reproducibility (default: None)
#         """
#         self.num_classes = num_classes
#         self.learning_rate = learning_rate
#         self.num_iterations = num_iterations
#         self.random_seed = random_seed
#         self.weights = None
 
#     def fit(self, X_train, y_train, X_val=None, y_val=None):
#         """
#         Train the Multinomial Logistic Regression model.
 
#         Parameters:
#         - X_train: Training feature data.
#         - y_train: Training labels.
#         - X_val: Validation feature data (optional).
#         - y_val: Validation labels (optional).
 
#         Returns:
#         - train_losses: List of training losses during iterations.
#         - train_accuracies: List of training accuracies during iterations.
#         - val_losses: List of validation losses during iterations (if validation data provided).
#         - val_accuracies: List of validation accuracies during iterations (if validation data provided).
#         """
#         # Add bias term to the training data
#         X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
 
#         # One-hot encode the labels
#         enc = OneHotEncoder(categories='auto')
#         y_train = enc.fit_transform(y_train.reshape(-1, 10)).toarray()
 
#         # Initialize weights with random values
#         np.random.seed(self.random_seed)
#         self.weights = np.random.randn(X_train_bias.shape[1], self.num_classes)
 
#         # Lists to store training and validation losses during training
#         train_losses = []
#         val_losses = []
#         train_accuracies = []
#         val_accuracies = []
 
#         epsilon = 1e-7  # Small constant to prevent division by zero in Adam optimizer
 
#         m_t = np.zeros(self.weights.shape)
#         v_t = np.zeros(self.weights.shape)
#         t = 0
 
#         for iteration in tqdm(range(self.num_iterations)):
#             # Calculate logits and softmax probabilities
#             logits = np.dot(X_train_bias, self.weights)
#             exp_logits = np.exp(logits)
#             softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
#             print(y_train.shape)
 
#             # Compute the cross-entropy loss
#             loss = -np.mean(y_train * np.log(softmax_probs))
 
#             # Calculate training accuracy
#             train_pred = np.argmax(softmax_probs, axis=1)
#             train_accuracy = accuracy_score(
#                 np.argmax(y_train, axis=1), train_pred)
 
#             train_accuracies.append(train_accuracy)
#             train_losses.append(loss)
 
#             # Calculate gradients
#             gradient = np.dot(X_train_bias.T, softmax_probs -
#                               y_train) / X_train_bias.shape[0]
 
#             t += 1
#             m_t = 0.9 * m_t + (1 - 0.9) * gradient
#             v_t = 0.999 * v_t + (1 - 0.999) * (gradient ** 2)
#             m_t_hat = m_t / (1 - 0.9 ** t)
#             v_t_hat = v_t / (1 - 0.999 ** t)
 
#             # Update weights using Adam optimizer
#             self.weights -= self.learning_rate * \
#                 m_t_hat / (np.sqrt(v_t_hat) + epsilon)
 
#         return train_losses, val_losses, train_accuracies, val_accuracies
 
#     def predict(self, X):
#         """
#         Make predictions using the trained model.
 
#         Parameters:
#         - X: Feature data for prediction.
 
#         Returns:
#         - predicted_class: Predicted class labels.
#         """
#         X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
#         logits = np.dot(X_bias, self.weights)
#         predicted_class = np.argmax(logits, axis=1)
#         return predicted_class