import numpy as np
import mlrose_hiive as mlrose
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import preprocess as prp
from joblib import dump, load



#preprocessing


#setup for airbnb Data

ab_data = prp.ab_data()
ab_data.clean()
ab_data.target = 'room_type'
ab_data.features = ab_data.all.columns[ab_data.all.columns != ab_data.target]
ab_data.init_model_data(target=ab_data.target,features=ab_data.features)
#TODO add OHE and SCALING to Preprocess

#BEST ESTIMATOR properties
# MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
#               beta_2=0.999, early_stopping=False, epsilon=1e-08,
#               hidden_layer_sizes=(5, 2, 2, 3), learning_rate='constant',
#               learning_rate_init=0.001, max_fun=15000, max_iter=200,
#               momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
#               power_t=0.5, random_state=None, shuffle=True, solver='adam',
#               tol=0.0001, validation_fraction=0.1, verbose=False,
#               warm_start=False)

#scale and fit
# https://mlrose.readthedocs.io/en/stable/source/tutorial3.html [SOURCE]
scaler = StandardScaler()
ab_data.x_train_scaled = scaler.fit_transform(ab_data.x_train)
ab_data.x_test_scaled = scaler.transform(ab_data.x_test)
# One hot encode target values
one_hot = OneHotEncoder()
ab_data.y_train_hot = one_hot.fit_transform(np.array(ab_data.y_train).reshape(-1, 1)).todense()
ab_data.y_test_hot = one_hot.transform(np.array(ab_data.y_test).reshape(-1, 1)).todense()



#EXISTING MODEL
NN_ab = load('clf_NN_abnb_final_0.8273.joblib') #TODO look into loss curve
NN_ab_trained = NN_ab._final_estimator.best_estimator_
NN_ab_trained.fit(ab_data.x_train,ab_data.y_train)
y_train_pred = NN_ab_trained.predict(ab_data.x_train)
y_train_pred_prob = NN_ab_trained.predict(ab_data.x_train)

y_test_pred = NN_ab_trained.predict(ab_data.x_test)
y_test_pred_prob = NN_ab_trained.predict_proba(ab_data.x_test)

NN_ab_trained_train_accur = accuracy_score(y_train_pred,ab_data.y_train)
NN_ab_trained_test_accur = accuracy_score(y_test_pred,ab_data.y_test)


print("NN_trained train acc",NN_ab_trained_train_accur)
print("NN_trained test acc",NN_ab_trained_test_accur)

print("NN_trained train acc",NN_ab_trained_train_accur)
print("NN_trained test acc",NN_ab_trained_test_accur)



#MLROSE
#Instantiate

NN_ab_randop = mlrose.NeuralNetwork(hidden_nodes = [5,2,2,3], activation = 'relu', \
                                 algorithm = 'simulated_annealing',max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.00001, \
                                 early_stopping = False, clip_max = 5, max_attempts = 2**30, \
                                 random_state = np.random.randint(0,2**30,1),schedule = mlrose.GeomDecay(init_temp=1.0, decay=0.25, min_temp=0.01))
#fit
NN_ab_randop.fit(ab_data.x_train_scaled, ab_data.y_train_hot)  #somehow this interfers with NN_ab_trained???????
# #
# #
# # #
# # # # Predict labels for train set and assess accuracy
ab_data.y_train_pred = NN_ab_randop.predict(ab_data.x_train_scaled)
ab_data.y_train_accuracy = accuracy_score(ab_data.y_train_hot, ab_data.y_train_pred)
ab_data.y_train_roc_accuracy = roc_auc_score(ab_data.y_train,ab_data.y_train_pred,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
# # # Predict labels for test set and assess accuracy
ab_data.y_test_pred = NN_ab_randop.predict(ab_data.x_test_scaled)
ab_data.y_test_accuracy = accuracy_score(ab_data.y_test_hot, ab_data.y_test_pred)
ab_data.y_test_roc_accuracy = roc_auc_score(ab_data.y_test,ab_data.y_test_pred,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
# #
print("SA NN train roc",ab_data.y_train_roc_accuracy)
print("SA NN test roc",ab_data.y_test_roc_accuracy)
print("SA NN train acc",ab_data.y_train_accuracy)
print("SA NN test acc",ab_data.y_test_accuracy)


NN_ab_randop = mlrose.NeuralNetwork(hidden_nodes = [5,2,2,3], activation = 'relu',\
                                 algorithm = 'random_hill_climb',max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.00001, \
                                 early_stopping = False, clip_max = 5, max_attempts = 2**30, \
                                 random_state = np.random.randint(0,2**30,1),schedule = mlrose.GeomDecay(init_temp=1.0, decay=0.25, min_temp=0.01))
#fit
NN_ab_randop.fit(ab_data.x_train_scaled, ab_data.y_train_hot)  #somehow this interfers with NN_ab_trained???????
# #
# #
# # #
# # # # Predict labels for train set and assess accuracy
ab_data.y_train_pred = NN_ab_randop.predict(ab_data.x_train_scaled)
ab_data.y_train_accuracy = accuracy_score(ab_data.y_train_hot, ab_data.y_train_pred)
ab_data.y_train_roc_accuracy = roc_auc_score(ab_data.y_train,ab_data.y_train_pred,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
# # # Predict labels for test set and assess accuracy
ab_data.y_test_pred = NN_ab_randop.predict(ab_data.x_test_scaled)
ab_data.y_test_accuracy = accuracy_score(ab_data.y_test_hot, ab_data.y_test_pred)
ab_data.y_test_roc_accuracy = roc_auc_score(ab_data.y_test,ab_data.y_test_pred,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
# #
print("RHC NN train roc",ab_data.y_train_roc_accuracy)
print("RHC NN test roc",ab_data.y_test_roc_accuracy)
print("RHC NN train acc",ab_data.y_train_accuracy)
print("RHC NN test acc",ab_data.y_test_accuracy)



NN_ab_randop = mlrose.NeuralNetwork(hidden_nodes = [5,2,2,3], activation = 'relu',pop_size=20,\
                                 algorithm = 'genetic_alg',max_iters = 100, \
                                 bias = True, is_classifier = True, learning_rate = 0.00001, \
                                 early_stopping = False, clip_max = 5, max_attempts = 20, \
                                 random_state = np.random.randint(0,2**30,1),schedule = mlrose.GeomDecay(init_temp=1.0, decay=0.25, min_temp=0.01))
#fit
NN_ab_randop.fit(ab_data.x_train_scaled, ab_data.y_train_hot)  #somehow this interfers with NN_ab_trained???????
# #
# #
# # #
# # # # Predict labels for train set and assess accuracy
ab_data.y_train_pred = NN_ab_randop.predict(ab_data.x_train_scaled)
ab_data.y_train_accuracy = accuracy_score(ab_data.y_train_hot, ab_data.y_train_pred)
ab_data.y_train_roc_accuracy = roc_auc_score(ab_data.y_train,ab_data.y_train_pred,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
# # # Predict labels for test set and assess accuracy
ab_data.y_test_pred = NN_ab_randop.predict(ab_data.x_test_scaled)
ab_data.y_test_accuracy = accuracy_score(ab_data.y_test_hot, ab_data.y_test_pred)
ab_data.y_test_roc_accuracy = roc_auc_score(ab_data.y_test,ab_data.y_test_pred,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
# #
print("RHC GA train roc",ab_data.y_train_roc_accuracy)
print("RHC GA test roc",ab_data.y_test_roc_accuracy)
print("RHC GA train acc",ab_data.y_train_accuracy)
print("RHC GA test acc",ab_data.y_test_accuracy)


