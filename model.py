import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

##################################################################################################

scana = pd.read_csv('scan_A.csv')
scansu = pd.read_csv('scan_sU.csv')
sparta = pd.read_csv('sparta.csv')
mqtt = pd.read_csv('mqtt_bruteforce.csv')
normal = pd.read_csv('normal.csv')

data = pd.concat([scana, scansu, sparta, mqtt, normal], axis=0)

##################################################################################################

data = data.drop(columns=["timestamp","src_ip","dst_ip", "src_port","dst_port"])

for column in data:
    if column not in ['protocol', 'is_attack']:
        clean_col = data[column].dropna()
        mean = clean_col.mean()
        data[column].fillna(mean, inplace=True)

##################################################################################################

#FEATURE SELECTION
X, y = data.drop(columns=["protocol","is_attack"]).values, data['is_attack'].values
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
importances = clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)

selected_indexes = [i+1 for i in range(len(importances)) if importances[i] >= 0.0001]
selected_indexes = selected_indexes+[0,len(data.columns)-1]

not_sel_col = [data.columns[i] for i in range(len(data.columns)) if i not in selected_indexes]
sel_col = [data.columns[i] for i in range(len(data.columns)) if i in selected_indexes]


data = data.drop(columns=not_sel_col)

data = data.sample(frac=1).reset_index(drop=True)

data['is_attack'].replace(0, 'no', inplace=True)
data['is_attack'].replace(1, 'yes', inplace=True)

##################################################################################################

#NORMALIZATION

def norm(x,xmin,xmax):
    if xmin != xmax:
        return (x-xmin)/(xmax-xmin)
    else:
        return xmin

for column in data:
    if column not in ['protocol', 'is_attack']:
        #put min at row 0 and max at row 1
        col_min = data[column].min()
        col_max = data[column].max()

        #apply normalization and substitute column
        data[column] = norm(data.loc[:, column], col_min, col_max)

##################################################################################################

X_train = data.loc[:int(0.8*len(data))]
X_test = data.loc[int(0.8*len(data)):(int(0.8*len(data))+int(0.1*len(data)))]
X_valid = data.loc[(int(0.8*len(data))+int(0.1*len(data))):]

train_data_file = "train_data.csv"
test_data_file = "test_data.csv"
valid_data_file = "test_data.csv"

X_train.to_csv(train_data_file, index=False, header=False)
X_test.to_csv(test_data_file, index=False, header=False)
X_valid.to_csv(valid_data_file, index=False, header=False)

##################################################################################################

CSV_HEADER = sel_col

# A list of the numerical feature names.
NUMERIC_FEATURE_NAMES = sel_col[1:len(sel_col)-1]

# A dictionary of the categorical features and their vocabulary.
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "protocol": sorted(list(X_train["protocol"].unique())),
    #"is_attack": sorted(list(X_train["is_attack"].unique())),
    #"Source": sorted(list(X_train["Source"].unique())),
    #"Syn": sorted(list(X_train["Syn"].unique())),
    #"Reset": sorted(list(X_train["Reset"].unique())),
    #"Acknowledgment": sorted(list(X_train["Acknowledgment"].unique())),
    #"Info": sorted(list(X_train["Info"].unique())),

    #"Destination": sorted(list(X_train["Destination"].unique())),
    #"Length": sorted(list(X_train["Length"].unique())),
    #"Classification": sorted(list(X_train["Classification"].unique())),
}
# Name of the column to be used as instances weight.
#WEIGHT_COLUMN_NAME = "fnlwgt"
# A list of the categorical feature names.
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
# A list of all the input features.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
# A list of column default values for each feature.
COLUMN_DEFAULTS = [
    #[0.0] if feature_name in NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME] else ["NA"]
    [0.0] if feature_name in NUMERIC_FEATURE_NAMES else ["NA"]
    for feature_name in CSV_HEADER
]
# The name of the target feature.
TARGET_FEATURE_NAME = "is_attack"
# A list of the labels of the target features.
TARGET_LABELS = [
                 'yes',
                 'no'
                 ]

##################################################################################################

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.2
BATCH_SIZE = 256
NUM_EPOCHS = 1

NUM_TRANSFORMER_BLOCKS = 10  # Number of transformer blocks.
NUM_HEADS = 4  # Number of attention heads.
EMBEDDING_DIMS = 16  # Embedding dimensions of the categorical features.
MLP_HIDDEN_UNITS_FACTORS = [
    2,
    1,
]  # MLP hidden layer units, as factors of the number of inputs.
NUM_MLP_BLOCKS = 2  # Number of MLP blocks in the baseline model.

##################################################################################################

target_label_lookup = layers.StringLookup(
    vocabulary=TARGET_LABELS, num_oov_indices=0,
    #mask_token=None,
    mask_token=None,
)


def prepare_example(features, target):
    target_index = target_label_lookup(target)
    #weights = features.pop(WEIGHT_COLUMN_NAME)
    return features, target_index#, weights

lookup_dict = {}
for feature_name in CATEGORICAL_FEATURE_NAMES:
    vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
    # Create a lookup to convert a string values to an integer indices.
    # Since we are not using a mask token, nor expecting any out of vocabulary
    # (oov) token, we set mask_token to None and num_oov_indices to 0.
    lookup = layers.StringLookup(
        vocabulary=vocabulary, num_oov_indices=0,
        mask_token=None
    )
    lookup_dict[feature_name] = lookup

def get_dataset_from_csv(csv_file_path, batch_size=128, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        na_value="?",
        shuffle=shuffle,
    ).map(prepare_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return dataset.cache()

##################################################################################################

def run_experiment(
    model,
    train_data_file,
    test_data_file,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.Precision(),
                 keras.metrics.Recall() ],
    )

    train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)
    validation_dataset = get_dataset_from_csv(test_data_file, batch_size)

    print("Start training the model...")
    history = model.fit(
        #print(train_dataset)
        train_dataset, epochs=num_epochs, validation_data=validation_dataset
    )
    print("Model training finished")

    _, accuracy, precision, recall = model.evaluate(validation_dataset, verbose=0)

    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")

    return history

##################################################################################################

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs

##################################################################################################

def encode_inputs(inputs, embedding_dims):

    encoded_categorical_feature_list = []
    numerical_feature_list = []

    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:

            # Get the vocabulary of the categorical feature.
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            if feature_name == 'protocol':
                df = pd.DataFrame(vocabulary)
                df.to_csv("vocab.csv")

            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int",
            )

            # Convert the string input values into integer indices.
            encoded_feature = lookup(inputs[feature_name])

            # Create an embedding layer with the specified dimensions.
            embedding = layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_dims
            )

            # Convert the index values to embedding representations.
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_categorical_feature_list.append(encoded_categorical_feature)

        else:

            # Use the numerical features as-is.
            numerical_feature = tf.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)

    return encoded_categorical_feature_list, numerical_feature_list

##################################################################################################

def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):

    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer),
        mlp_layers.append(layers.Dense(units, activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(mlp_layers, name=name)

##################################################################################################

def create_tabtransformer_classifier(
    num_transformer_blocks,
    num_heads,
    embedding_dims,
    mlp_hidden_units_factors,
    dropout_rate,
    use_column_embedding=False,
):

    # Create model inputs.
    inputs = create_model_inputs()
    # encode features.
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, embedding_dims
    )
    # Stack categorical feature embeddings for the Tansformer.
    encoded_categorical_features = tf.stack(encoded_categorical_feature_list, axis=1)
    # Concatenate numerical features.
    numerical_features = layers.concatenate(numerical_feature_list)

    # Add column embedding to categorical feature embeddings.
    if use_column_embedding:
        num_columns = encoded_categorical_features.shape[1]
        column_embedding = layers.Embedding(
            input_dim=num_columns, output_dim=embedding_dims
        )
        column_indices = tf.range(start=0, limit=num_columns, delta=1)
        encoded_categorical_features = encoded_categorical_features + column_embedding(
            column_indices
        )

    # Create multiple layers of the Transformer block.
    for block_idx in range(num_transformer_blocks):
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f"multihead_attention_{block_idx}",
        )(encoded_categorical_features, encoded_categorical_features)
        # Skip connection 1.
        x = layers.Add(name=f"skip_connection1_{block_idx}")(
            [attention_output, encoded_categorical_features]
        )
        # Layer normalization 1.
        x = layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)
        # Feedforward.
        feedforward_output = create_mlp(
            hidden_units=[embedding_dims],
            dropout_rate=dropout_rate,
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=1e-6),
            name=f"feedforward_{block_idx}",
        )(x)
        # Skip connection 2.
        x = layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
        # Layer normalization 2.
        encoded_categorical_features = layers.LayerNormalization(
            name=f"layer_norm2_{block_idx}", epsilon=1e-6
        )(x)

    # Flatten the "contextualized" embeddings of the categorical features.
    categorical_features = layers.Flatten()(encoded_categorical_features)
    # Apply layer normalization to the numerical features.
    numerical_features = layers.LayerNormalization(epsilon=1e-6)(numerical_features)
    # Prepare the input for the final MLP block.
    features = layers.concatenate([categorical_features, numerical_features])

    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="MLP",
    )(features)

    # Add a sigmoid as a binary classifer.
    outputs = layers.Dense(units=1, activation="softmax", name="softmax")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


tabtransformer_model = create_tabtransformer_classifier(
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    num_heads=NUM_HEADS,
    embedding_dims=EMBEDDING_DIMS,
    mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
    dropout_rate=DROPOUT_RATE,
)

print("Total model weights:", tabtransformer_model.count_params())
keras.utils.plot_model(tabtransformer_model, show_shapes=True, rankdir="LR")

##################################################################################################

history = run_experiment(
    model=tabtransformer_model,
    train_data_file=train_data_file,
    test_data_file=test_data_file,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
)











