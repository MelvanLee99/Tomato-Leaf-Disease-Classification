import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

import os

LOG_DIR = './logs/'

def one_hot_encode(image, label, num_classes):
    return image, tf.one_hot(label, depth=num_classes)

def preprocess(image, label):
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0 
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def get_class_distribution_data(y_data, name, class_names=None):
    if y_data.size == 0:
        print(f'\n--- {name} class distribution ---')
        print('  (No samples in this set)')
        return pd.DataFrame(columns=['Dataset', 'Class ID', 'Class Name', 'Count', 'Percentage'])

    if y_data.ndim > 1 and y_data.shape[1] > 1:
        y_integer_labels = np.argmax(y_data, axis=-1)
    else:
        y_integer_labels = np.squeeze(y_data)

    unique_classes, counts = np.unique(y_integer_labels, return_counts=True)
    total_samples = len(y_integer_labels)

    data = []
    for i, c_id in enumerate(unique_classes):
        c_id = int(c_id)
        percentage = (counts[i] / total_samples) * 100
        class_label = class_names[c_id] if class_names and c_id < len(class_names) else f'class_{c_id}'
        data.append({
            'data_subset': name,
            'class_id': c_id,
            'class_name': class_label,
            'count': counts[i],
            'percentage': round(percentage, 4)
        })
    
    return data

def split_data(ds, is_binary, class_names=None):
    images = []
    labels = []
    for image, label in ds:
        images.append(image.numpy())
        labels.append(label.numpy())
    # del ds

    X = np.array(images)
    y = np.array(labels)
    del images, labels, image, label

    if is_binary:
        y = np.expand_dims(y, -1)
    print(X.shape, y.shape)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=101, stratify=y
    )
    del X, y

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=101, stratify=y_temp
    )
    del X_temp, y_temp

    print(f'\nOverall Split Sizes:')
    print(f'Train set shape: {X_train.shape}, {y_train.shape}')
    print(f'Validation set shape: {X_val.shape}, {y_val.shape}')
    print(f'Test set shape: {X_test.shape}, {y_test.shape}')

    dist = []
    train_dist = get_class_distribution_data(y_train, 'train', class_names)
    val_dist = get_class_distribution_data(y_val, 'validation', class_names)
    test_dist = get_class_distribution_data(y_test, 'test', class_names)
    dist.extend(train_dist)
    dist.extend(val_dist)
    dist.extend(test_dist)
    dist = pd.DataFrame(dist)
    
    os.makedirs(LOG_DIR, exist_ok=True)
    num_classes = len(class_names)
    if num_classes == 2:
        filename = 'class_distribution_binary'
    elif num_classes == 5:
        filename = 'class_distribution_quinary'
    elif num_classes == 10:
        filename = 'class_distribution_main'
    out_file = os.path.join(LOG_DIR, f'{filename}.csv')
    dist.to_csv(out_file, index=False)
    print(f'\nClass distribution data has been saved to: {os.path.relpath(LOG_DIR)}')
    del train_dist, val_dist, test_dist, dist

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    del X_train, y_train
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    del X_val, y_val
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    del X_test, y_test

    return train_ds, val_ds, test_ds

def predict(model, test_ds, is_binary):
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images)
        if is_binary:
            y_true.extend(labels[:, 0].numpy())
            pred_labels = preds[:, 0] > 0.5
        else:
            y_true.extend(np.argmax(labels.numpy(), axis=1)) # one-hot encoded labels
            pred_labels = np.argmax(preds, axis=1)
        y_pred.extend(pred_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return y_true, y_pred