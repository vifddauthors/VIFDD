"""
This script performs 3-fold cross-validation on a list of pre-trained models using a given dataset.
It trains each model on a subset of the dataset and evaluates its performance on the validation set.
The script outputs the accuracy, precision, recall, F1 score, and AUC score for each fold and the average
and standard deviation of these metrics across all folds.
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Perform 5-fold cross-validation on pre-trained models.')
parser.add_argument('--dataset_path', type=str, help='Path to the dataset directory', required=True)
args = parser.parse_args()

# Set the paths to your dataset
data_directory = args.dataset_path

# Set hyperparameters
epochs = 10
batch_size = 64
optimizer = "adam"
loss = "binary_crossentropy"


# Define the common model architecture
def create_model(base_model):
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model


# Define the list of pre-trained models
pretrained_models = [
    tf.keras.applications.MobileNetV2,
    tf.keras.applications.EfficientNetV2S,
    tf.keras.applications.ResNet50,
    tf.keras.applications.ResNet101,
    tf.keras.applications.Xception,
    tf.keras.applications.InceptionV3,
    tf.keras.applications.DenseNet121,
    tf.keras.applications.NASNetMobile,
    tf.keras.applications.InceptionResNetV2,
]

# Load dataset using tf.keras.utils.image_dataset_from_directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    shuffle=True,
    batch_size=batch_size,
    image_size=(224, 224),
    seed=42,
    label_mode="binary",
)

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

num_samples = len(train_dataset.file_paths)

# Perform 5-fold cross-validation
kf = KFold(n_splits=3, shuffle=True)
indices = np.arange(num_samples)
output_file = "output.txt"

with open(output_file, "w") as f:
    for model_class in pretrained_models:
        f.write(f"Results for {model_class.__name__}:\n")

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        auc_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"Training fold {fold + 1} for {model_class.__name__}")

            # Create the pre-trained base model
            base_model = model_class(
                weights=None, include_top=False, input_shape=(224, 224, 3)
            )

            # Create the full model
            model = create_model(base_model)

            # Compile the model
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=["accuracy", "precision", "recall", "auc"],
            )
            train_fold_dataset = None
            for i in range(3):
                if i != fold:
                    if train_fold_dataset == None:
                        train_fold_dataset = train_dataset.shard(num_shards=5, index=i)
                    else:
                        train_fold_dataset = train_fold_dataset.concatenate(
                            train_dataset.shard(num_shards=3, index=i)
                        )

            val_dataset = train_dataset.shard(num_shards=3, index=fold)

            # Train the model
            model.fit(
                train_fold_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=[early_stopping],
            )

            # Evaluate the model on the validation set
            eval_metrics = model.evaluate(
                test_dataset, steps=val_steps, return_dict=True
            )

            accuracy = eval_metrics["accuracy"]
            precision = eval_metrics["precision"]
            recall = eval_metrics["recall"]
            f1 = (2 * precision * recall) / (precision + recall)
            auc = eval_metrics["auc"]

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            auc_scores.append(auc)

            f.write(
                f"Fold {fold + 1} - "
                f"Accuracy: {accuracy}, "
                f"Precision: {precision}, "
                f"Recall: {recall}, "
                f"F1 Score: {f1}, "
                f"AUC Score: {auc}\n"
            )

            # Clear session to release GPU memory
            tf.keras.backend.clear_session()

        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        avg_precision = sum(precision_scores) / len(precision_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)
        avg_auc = sum(auc_scores) / len(auc_scores)

        std_accuracy = (
            sum((x - avg_accuracy) ** 2 for x in accuracy_scores) / len(accuracy_scores)
        ) ** 0.5
        std_precision = (
            sum((x - avg_precision) ** 2 for x in precision_scores)
            / len(precision_scores)
        ) ** 0.5
        std_recall = (
            sum((x - avg_recall) ** 2 for x in recall_scores) / len(recall_scores)
        ) ** 0.5
        std_f1 = (sum((x - avg_f1) ** 2 for x in f1_scores) / len(f1_scores)) ** 0.5
        std_auc = (sum((x - avg_auc) ** 2 for x in auc_scores) / len(auc_scores)) ** 0.5

        f.write(
            f"Average - "
            f"Accuracy: {avg_accuracy}, "
            f"Precision: {avg_precision}, "
            f"Recall: {avg_recall}, "
            f"F1 Score: {avg_f1}, "
            f"AUC Score: {avg_auc}\n"
        )
        f.write(
            f"Standard Deviation - "
            f"Accuracy: {std_accuracy}, "
            f"Precision: {std_precision}, "
            f"Recall: {std_recall}, "
            f"F1 Score: {std_f1}, "
            f"AUC Score: {std_auc}\n\n"
        )
