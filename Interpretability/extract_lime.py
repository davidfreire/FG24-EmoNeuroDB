import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from lime.lime_tabular import LimeTabularExplainer

import pickle
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
import random

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import clear_session
from sklearn.decomposition import PCA
import gc
from keras import backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')


import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Reshape, LSTM, Flatten, Concatenate, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import joblib
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder

class Classifier:
    def __init__(self, id_experiment):
        self.id_experiment = id_experiment

    def evaluate_challenge_model(self, id_model, train, val, bucket_div):
        print('Starting training...')

        X_train_left = train[0]
        X_train_right = train[1]
        y_train = train[2]
        X_val_left = val[0]
        X_val_right = val[1]
        y_val = val[2]

        model = LSTM_2branches(X_train_left)

        # Display the model summary
        model.summary()
        weights_file = 'emoneuro_challenge/weights'
        if not os.path.exists(weights_file):
            os.makedirs(weights_file)

        weights_filename = os.path.join(weights_file, self.id_experiment+ '_'+id_model+'_bckt_'+str(bucket_div)+'.weights.h5')
        print(weights_filename)

        ckpt = ModelCheckpoint(weights_filename,
                               save_best_only=True, save_weights_only=True,
                               monitor='val_accuracy', verbose=0, mode='max')

        earlystopper = EarlyStopping(monitor='val_loss', patience=20)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

        # Model fitting
        model.fit([X_train_left, X_train_right], y_train, batch_size=64, validation_data=([X_val_left, X_val_right], y_val),
                  epochs=200, verbose=1,
                  callbacks=[ckpt, earlystopper, reduce_lr])

        # Load the best model
        model.load_weights(weights_filename)

        print('Evaluating best on train data...')
        eval_train = model.evaluate([X_train_left, X_train_right], y_train)

        print('Evaluating best on val data...')
        eval_val = model.evaluate([X_val_left, X_val_right], y_val)

        return eval_val, model

    def run_restored_model(self, id_model, weights_filename, test, test_filenames, bucket_div):
        print('Starting restoring...')

        X_test = test
        model = LSTM_2branches(X_test[0])

        # Load the best model
        model.load_weights(weights_filename)

        # Display the model summary
        model.summary()

        label_encoder = joblib.load('label_encoder.joblib')

        eval_path = 'emoneuro_challenge/evaluation_from_saved_model'
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)

        print('Evaluating dataframes...')
        predictions = model.predict([X_test[0], X_test[1]])
        predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        result_df = pd.DataFrame({'filename': test_filenames, 'class': predicted_classes})
        print(result_df['class'].value_counts())
        result_df.to_csv(os.path.join(eval_path, 'eval_'+id_model+'bckt_'+str(bucket_div)+'.csv'), index=False)
        return model # variacion
        del model
        gc.collect()
    def generate_evaluation_file(self, id_test, X_test, test_filenames, best_model, bucket_div):

        label_encoder = joblib.load('label_encoder.joblib')

        eval_path = 'emoneuro_challenge/evaluation_from_trained_model'
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)

        # Use the evaluate_dataframes function
        print('Evaluating dataframes...')

        # Make predictions on the new data
        predictions = best_model.predict([X_test[0], X_test[1]])

        # Convert predicted class indices back to string labels
        predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

        # Create a DataFrame with 'video_ID' and 'predicted_class'
        result_df = pd.DataFrame({'filename': test_filenames, 'class': predicted_classes})

        print(result_df['class'].value_counts())

        result_df.to_csv(os.path.join(eval_path, id_test+'bckt_'+str(bucket_div)+'.csv'), index=False)

    def explain_with_lime(self, model, X_sample):
        # Concatenate the two branches into a single 2D array for LIME
        input_data = np.concatenate([X_sample[0].reshape(-1), X_sample[1].reshape(-1)])

        # Define the LimeTabularExplainer
        explainer = LimeTabularExplainer(
            np.array([input_data]),  # Example input shape
            feature_names=[f'feature_{i}' for i in range(input_data.size)],
            class_names=[str(i) for i in range(6)],
            discretize_continuous=False
        )

        # Define a custom predict function that splits the input for two branches
        def predict_fn(instance):
            # Split the instance back into two branches
            branch1 = instance[:, :X_sample[0].size].reshape(-1, X_sample[0].shape[0], X_sample[0].shape[1])
            branch2 = instance[:, X_sample[0].size:].reshape(-1, X_sample[1].shape[0], X_sample[1].shape[1])
            return model.predict([branch1, branch2])

        # Explain the instance with LIME
        exp = explainer.explain_instance(input_data, predict_fn, num_features=11)
        exp.show_in_notebook(show_table=True)
        return exp

# Define the model architecture outside the class
def LSTM_2branches(input_data):
    input_layers = []
    processed_outputs = []

    inp = Input(shape=(input_data.shape[1], input_data.shape[2]), name='input_signal_left')
    input_layers.append(inp)
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inp)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Reshape((x.shape[1], x.shape[2]))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)

    processed_outputs.append(x)

    inp2 = Input(shape=(input_data.shape[1], input_data.shape[2]), name='input_signal_right')
    input_layers.append(inp2)
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inp2)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Reshape((x.shape[1], x.shape[2]))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)

    processed_outputs.append(x)
    concatenated_out = Concatenate()(processed_outputs)

    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(concatenated_out)
    x = Dense(6, activation='softmax')(x)

    model = Model(inputs=input_layers, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Function to explain an instance with LIME
def explain_with_lime(model, X_sample, feat_names, ground_truth_label):
    # Concatenate the two branches into a single 2D array for LIME
    X_sample = (X_sample[0].reshape(4488, 11), X_sample[1].reshape(4488, 11))

    input_data = np.concatenate([X_sample[0].reshape(-1), X_sample[1].reshape(-1)])

    
    # Define the LimeTabularExplainer
    explainer = LimeTabularExplainer(
        np.array([input_data]),  # Example input shape
        feature_names=[f'feature_{i}' for i in range(input_data.size)],
        class_names=[str(i) for i in range(6)],
        discretize_continuous=False
    )
    
    
    # Custom predict function that reshapes the input for the model
    def predict_fn(instance):
        # Split the instance back into two branches for the model
        branch1 = instance[:, :X_sample[0].size].reshape(-1, X_sample[0].shape[0], X_sample[0].shape[1])
        branch2 = instance[:, X_sample[0].size:].reshape(-1, X_sample[1].shape[0], X_sample[1].shape[1])
        return model.predict([branch1, branch2])

    # Explain the instance with LIME
    exp = explainer.explain_instance(input_data, predict_fn, labels=[ground_truth_label], num_features=input_data.size, num_samples=5000)

    avg_imp_left_stream = np.zeros(11)
    avg_imp_right_stream = np.zeros(11)
    
    feature_importances = exp.local_exp[ground_truth_label]
    
    for feature_index, importance_value in feature_importances:
        if feature_index < 4488*11: # left stream
            original_feat_idx = feature_index % 11
            avg_imp_left_stream[original_feat_idx] +=  importance_value/4488
        else:
            original_feat_idx = (feature_index - 4488*11) % 11
            avg_imp_right_stream[original_feat_idx] +=  importance_value/4488
            
    avg_imp_streams = np.concatenate([avg_imp_left_stream, avg_imp_right_stream])
    # Collect LIME details
    feature_imp = {f'{feat}_imp': imp for (feat, imp) in zip(feat_names, avg_imp_streams)}
    # exp_map is the same as local_exp, with a different format
    #local_exp_map = {f'exp_map_{i}': importance for i, importance in exp.as_map()[ground_truth_label]}
    prediction_probabilities = {f'{label_encoder.classes_[i]}_proba': prob for i, prob in enumerate(exp.predict_proba)}

    return feature_imp, exp.score, prediction_probabilities



outpath = 'emoneuro_challenge/data/processed/stage_1/'
df_train_val = pd.read_csv(os.path.join(outpath, 'train_val.csv'))

# Loading a pre-trained solution

datapath = 'emoneuro_challenge/data/processed/stage_2/dataset_1buckets.pkl'
with open(datapath, 'rb') as f:
    data = pickle.load(f)



experiment_id = 'Dev_LSTM'


clf = Classifier(experiment_id)

model_id = 'model_1_2branch'


train = data['train']
val = data['val']
test = data['test']
pca_data=data['pca_data']
bucket_div = data['bucket_div']
train_IDs = data['train_IDs']
val_IDs = data['val_IDs']
test_IDs = data['test_IDs']
test_filenames = data['test_filenames']


print('2 branches with {0} Buckets'.format(bucket_div))
print('Train set: {0}'.format(train_IDs))
print('Val set: {0}'.format(val_IDs))
print('Test set: {0}'.format(test_IDs))

print('Model: ', model_id)

weights_filename = os.path.join('emoneuro_challenge/weights', 'Dev_LSTM_2branches.h5')
my_model = clf.run_restored_model(model_id, weights_filename, test, test_filenames, bucket_div)


# Load the label encoder to decode ground truth and predictions
label_encoder = joblib.load('label_encoder.joblib')

feat_names = ['left_Fp1', 'left_F7', 'left_C3', 'left_P3', 'left_O1',
              'left_F3', 'left_T3', 'left_T5', 'left_Fz', 'left_Cz', 'left_A1', 
              'right_Fp2', 'right_F8', 'right_C4', 'right_P4', 'right_O2',
              'right_F4', 'right_T4', 'right_T6', 'right_Fz', 'right_Cz', 'right_A2']

users = df_train_val[df_train_val.train==2].filename.unique()

# Initialize lists to store the information for the DataFrame
user_ids = []
ground_truths = []
predicted_labels = []
lime_feature_importance = []
lime_score = []
lime_prediction_proba = []

for idx, user_id in enumerate(users):
    print('Procesando: ', user_id, '(', idx, '/', len(users), ')')
    X_sample = (val[0][idx:idx+1], val[1][idx:idx+1])  # Get the input for this user
    true_label = label_encoder.inverse_transform([np.argmax(val[2][idx])])[0]  # Ground truth label
    prediction = label_encoder.inverse_transform([np.argmax(my_model.predict([X_sample[0], X_sample[1]]))])[0]  # Predicted label

    # Run LIME explanation
    feature_imp, exp_score, prediction_probabilities = explain_with_lime(
        my_model, X_sample, feat_names, ground_truth_label=np.argmax(val[2][idx])
    )

    # Append information to lists for DataFrame creation
    user_ids.append(user_id)
    ground_truths.append(true_label)
    predicted_labels.append(prediction)
    lime_feature_importance.append(feature_imp)
    lime_score.append(exp_score)
    lime_prediction_proba.append(prediction_probabilities)

    if idx % 10:
        # Combine all data into a DataFrame
        df_results = pd.DataFrame({
            'user_id': user_ids,
            'ground_truth': ground_truths,
            'predicted_label': predicted_labels,
            'lime_feature_importance': lime_feature_importance,
            'lime_score': lime_score,
            'lime_prediction_proba': lime_prediction_proba
        })

        # Split the dictionary columns into individual columns
        df_results = pd.concat([df_results.drop(columns=['lime_feature_importance', 'lime_prediction_proba']),
                                df_results['lime_feature_importance'].apply(pd.Series),
                                df_results['lime_prediction_proba'].apply(pd.Series)], axis=1)


        df_results.to_csv('Lime_on_validation.csv', index=False)


# Combine all data into a DataFrame
df_results = pd.DataFrame({
    'user_id': user_ids,
    'ground_truth': ground_truths,
    'predicted_label': predicted_labels,
    'lime_feature_importance': lime_feature_importance,
    'lime_score': lime_score,
    'lime_prediction_proba': lime_prediction_proba
})

# Split the dictionary columns into individual columns
df_results = pd.concat([df_results.drop(columns=['lime_feature_importance', 'lime_prediction_proba']),
                        df_results['lime_feature_importance'].apply(pd.Series),
                        df_results['lime_prediction_proba'].apply(pd.Series)], axis=1)


df_results.to_csv('Lime_on_validation.csv', index=False)


