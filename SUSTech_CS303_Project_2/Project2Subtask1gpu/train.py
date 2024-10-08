# %%
import util
from SoftmaxRegression import SoftmaxRegression
from ANN import *
from tensorboardX import SummaryWriter
import time
from ANN2 import *
from ANN_opt import *
import numpy as npp
import cupy as np

# %% [markdown]
# ## 1. Data Loading 

# %%
classification_train_data = util.load_data("./data/classification_train_data.pkl")
classification_train_label = util.load_data("./data/classification_train_label.pkl")
classification_test_data = util.load_data("./data/classification_test_data.pkl")

# %% [markdown]
# ## 2. Data Exploration

# %%
print("Classification Train Data Shape:", classification_train_data.shape)
print("Classification Train Label Shape:", classification_train_label.shape)
print("Classification Test Data Shape:", classification_test_data.shape)

# %% [markdown]
# ## 3. Data Preprocessing

# %%
# remove index column
train_data_index = classification_train_data[:, 0]
train_label_index = classification_train_label[:, 0]
test_data_index = classification_test_data[:, 0]
classification_train_data = classification_train_data[:, 1:]
classification_train_label = classification_train_label[:, 1:].reshape(-1)
classification_test_data = classification_test_data[:, 1:]

# %%
classification_train_data.shape, classification_train_label.shape, classification_test_data.shape

# %%
train_data_index.shape, train_label_index.shape, test_data_index.shape

# %%
# normalization

# calculate the mean and standard deviation of each column
mean = np.mean(classification_train_data, axis=0)
std_dev = np.std(classification_train_data, axis=0)

# Z-Score normalizes each column
classification_train_data = (classification_train_data - mean) / std_dev
classification_test_data = (classification_test_data - mean) / std_dev

# %%
# label one-hot encoding
num_classes =  10 
classification_train_label = np.eye(num_classes)[classification_train_label]
print("train label shape:", classification_train_label.shape)

# %% [markdown]
# ## 4. Dataset Splitting

# %%
# divide the data set into training set and validation set
train_ratio = 0.8
seed = 123
(train_data, train_labels), (validation_data, validation_labels) = util.split_train_validation(
    classification_train_data, classification_train_label,
    train_ratio=train_ratio, random_seed=seed
    )

# %%

train_set = (train_data, train_labels)
val_set = (validation_data, validation_labels)
train_data.shape, train_labels.shape, validation_data.shape, validation_labels.shape


# %% [markdown]
# # 5. Model

# %%
ann = MyANN(n_inputs=256,n_hidden=24,n_outputs=10)
ann2 = NeuralNetwork(n_inputs=256,n_hidden=8,n_outputs=10,dropout_rate=0.2)
ann_opt = MyANN_opt(n_inputs=256,n_hidden=16,n_outputs=10,dropout_rate=0.2)


# %%
train_wr = SummaryWriter(log_dir='logs/Train')
val_wr = SummaryWriter(log_dir='logs/Val')

# %% [markdown]
# ## 6. Train 

# %%
import os
def random_shuffle(data,label):
    randnum = np.int32(np.random.randint(0, 1234).get())
    np.random.seed(randnum)
    np.random.shuffle(data)
    np.random.seed(randnum)
    np.random.shuffle(label)
    return data,label

def train_network(model:NeuralNetwork, train_set:tuple, val_set:tuple, opt:My_Opt, n_epoch, save_path, loss_func):
    train_size=len(train_set[0])
    val_size=len(val_set[0])
    train_x, train_y = train_set
    val_x, val_y = val_set
    best_val = 0.0
    for epoch in range(n_epoch):
        train_pred=[]
        train_loss = 0
        learning_rate = opt.get_learning_rate()
        train_x, train_y = random_shuffle(train_x, train_y)
        val_x, val_y = random_shuffle(val_x, val_y)
        for index in tqdm(range(train_size)):
            row, expected = train_x[index], train_y[index]
            outputs = model.forward(row)
            train_pred.append(outputs)
            train_loss += loss_func(outputs, expected)
            model.backward(row, expected, outputs, learning_rate)
        opt.step()
        train_accuracy = accuracy_score(np.argmax(train_y, axis=1), np.argmax(train_pred, axis=1))
        train_wr.add_scalar('accuracy', train_accuracy, epoch)
        train_wr.add_scalar('loss', train_loss, epoch)


        val_pred=[]
        val_loss = 0
        for index in tqdm(range(val_size)):
            row, expected = val_x[index], val_y[index]
            outputs = model.predict(row)
            val_pred.append(outputs)
            val_loss += loss_func(outputs, expected)
        val_accuracy = accuracy_score(np.argmax(val_y, axis=1), np.argmax(val_pred, axis=1))
        val_wr.add_scalar('accuracy', val_accuracy, epoch)
        val_wr.add_scalar('loss', val_loss, epoch)
        os.system('clear')
        if (val_accuracy > best_val):
            best_val = val_accuracy
            util.save_data(save_path, model)
            util.save_data(save_path+'.opt', opt)
            
        print('\r', end='', flush=True)
        print('>epoch=%d, learning_rate=%.3f, train_acc=%.4f, val_acc=%.4f' % (epoch, learning_rate, train_accuracy,val_accuracy))

# %%
loss_func=cross_entropy_loss
my_opt = My_Opt(alpha=0.5, beta1=0.9, beta2=0.999, epsilon=1e-8)

# train_network(model=ann,train_set=train_set, val_set=val_set, opt=my_opt,
#               n_epoch=30, save_path='./pths/model.pth', loss_func=cross_entropy_loss)


# ann_opt = util.load_data('./pths/model2.pth')
# my_opt = util.load_data('./pths/model2.pth.opt')
# %%
train_network(model=ann_opt,train_set=train_set, val_set=val_set, opt=my_opt, n_epoch=30, save_path='./pths/model2.pth', loss_func=cross_entropy_loss)


# %% [markdown]
# ## 7. Predict

# %%
# test_label_predict = ann.predict(classification_test_data)

# # %%
# # merge index and corresponding classification results 
# submit_data = np.hstack((
#     test_data_index.reshape(-1, 1),
#     test_label_predict.reshape(-1, 1)
#     ))

# # %%
# submit_data.shape

# # %%

# util.save_data('./classification_results.pkl', submit_data)


