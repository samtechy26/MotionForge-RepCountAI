import pickle
from sklearn.neural_network import MLPClassifier
from model_features import (
    feature_set_4,
    X_train,
    y_train,
)

selected_train_X = X_train[feature_set_4]
hidden_layer_sizes = (100,)
max_iter = 2000
activation = "logistic"
alpha = 0.0001
learning_rate = "adaptive"

model = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    max_iter=max_iter,
    learning_rate=learning_rate,
    alpha=alpha,
)

# Fit the model
model.fit(
    selected_train_X,
    y_train.values.ravel(),
)

export_model = open("../../models/nn_classifier,pkl", "wb")
pickle.dump(model, export_model)
export_model.close()
