
#!/usr/bin/python

# Train a multi-layer perceptron neural network random net
# A neural network to classify images of handwritten digits
# Tensorflow, GPU-Acceleration, MLP, Random Forest

import pandas as pd
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, GaussianNoise, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the handwritten digits dataset from sklearn
data = load_digits()

# Define label encoder
le = LabelEncoder()
le.fit(data.target)
n_digits = len(data.target_names)

# Separate a 10% hold out data set for future predictions
# After the random forest is trained
train_x, new_x, train_y, new_y = train_test_split(
        data.data,
        data.target,
        stratify=data.target,
        test_size=0.1)

train_x = pd.DataFrame(train_x)
new_x = pd.DataFrame(new_x)

# Data set totals
print("Training Samples =", str(len(train_x)))
print("Hold Out Samples =", str(len(new_x)))

# Define function to evaluate a single MLP model
def train_model(rf_train_x, rf_train_y, rf_test_x, rf_test_y, n_features):
    
    # Encode targets
    rf_train_y = to_categorical(rf_train_y, num_classes=n_digits)
    rf_test_y = to_categorical(rf_test_y, num_classes=n_digits)
    
    # Define sequential model
    model = Sequential()
    model.add(Dense(50,input_dim=n_features))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    model.add(Dense(15))
    model.add(Activation('relu'))
    model.add(GaussianNoise(0.10))
    model.add(Dense(n_digits))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
    
    # Define callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=3, verbose=1, restore_best_weights=True)
    
    # Fit model
    model.fit(rf_train_x.values, rf_train_y, epochs=100, batch_size=1000, 
              verbose=1, validation_data=(rf_test_x.values,rf_test_y),
              callbacks=[early_stop])
    
    # Evaluate model
    _, test_acc = model.evaluate(rf_test_x.values, rf_test_y, verbose=0)
    return model, test_acc

# Execute multiple train/test splits
n_splits = 31
scores, models, cols = list(), list(), list()
for i in range(n_splits):
    
    # Select a random subset of features
    # Take the standard square root
    n = int(round(np.sqrt(train_x.shape[1])))
    rf_train_x = train_x.sample(n,axis=1)
    n_features = rf_train_x.shape[1]
    
    # Split data with 80% random bagging fraction
    rf_train_x, rf_test_x, rf_train_y, rf_test_y = train_test_split(rf_train_x, train_y, test_size=0.20)
    
    # Train the model
    model, test_acc = train_model(rf_train_x, rf_train_y, rf_test_x, rf_test_y, n_features)
    
    print('Round %.0f Complete > %.3f' % (i+1,test_acc))
    scores.append(test_acc)
    models.append(model)
    cols.append(rf_train_x.columns)

# Summarize expected performance from a single weak learner
print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# evaluate a specific number of members in an ensemble
def evaluate_n_members(models, n_models, new_x, new_y):
    
    # Select a subset of models
    # Isolate relevant columns
    subset = models[:n_models]
    subset_cols = cols[:n_models]
    
    # Make predictions
    yhats = list()
    for i in range(0,len(subset)):        
        new_x_subset = new_x[subset_cols[i]]
        yhati = subset[i].predict(new_x_subset.values)
        yhati = np.array(yhati)
        yhats.append(yhati)
    
    # Compute bootstrapped accuracy
    summed = np.sum(yhats, axis=0)
    yhat = np.argmax(summed, axis=1)
    return accuracy_score(new_y, yhat)

# Evaluate the random net on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1,n_splits+1):
    
    # Compute ensemble score
    ensemble_score = evaluate_n_members(models, i, new_x, new_y)
    
    # Compute single score
    new_y_enc = to_categorical(new_y, num_classes=n_digits)
    new_subset = new_x[cols[i-1]]
    _, single_score = models[i-1].evaluate(new_subset.values, new_y_enc, verbose=0)
    
    # Print Results
    print('> %d: single=%.3f, random net=%.3f' % (i, single_score, ensemble_score))
    ensemble_scores.append(ensemble_score)
    single_scores.append(single_score)

# Plot individual scores vs. ensembled scores
print('Accuracy %.3f (%.3f)' % (np.mean(single_scores), np.std(single_scores)))
x_axis = [i for i in range(1, n_splits+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.title('Classification Performance\nRandom Net v. Individual Weak Learners')
pyplot.legend(['Weak Learners','Random Net'])
pyplot.show()

