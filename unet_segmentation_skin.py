import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import img_to_array, load_img
from skimage.transform import resize
from tqdm import tqdm
import time
import gc
from tensorflow.keras import backend as K
import datetime
import random

# -----------------------
# Dataset paths
# -----------------------
image_dir = r"C:\Users\PATH\Dataset-ISIC-2017-H\Images"
mask_dir = r"C:\Users\PATH\Dataset-ISIC-2017-H\Masks"

# -----------------------
# Grid search combinations list
# -----------------------
list_num_steps = [4]  # number of U-Net steps   [3, 4, 5]
list_base_filters = [16]  # starting filters   [4, 8, 16, 32, 64]

# -----------------------
# Loading images
# -----------------------
image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

# -----------------------
# Random reproducible subset selection
# -----------------------
MAX_IMAGES = 500  # number of images to use: 2750
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
random.seed(42)     # fixed seed for reproducibility

combined = list(zip(image_filenames, mask_filenames))
random.shuffle(combined)

image_filenames, mask_filenames = zip(*combined)
image_filenames = list(image_filenames[:MAX_IMAGES])
mask_filenames = list(mask_filenames[:MAX_IMAGES])

num_images = len(image_filenames)
X = np.zeros((num_images, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y = np.zeros((num_images, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

for i, (img_name, mask_name) in enumerate(
        tqdm(zip(image_filenames, mask_filenames), total=num_images, desc="Loading images")):
    img_array = img_to_array(load_img(os.path.join(image_dir, img_name)))
    mask_array = img_to_array(load_img(os.path.join(mask_dir, mask_name), color_mode="grayscale"))

    img_array = resize(img_array, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True)
    mask_array = resize(mask_array, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True, order=0, anti_aliasing=False)

    if mask_array.ndim == 2:
        mask_array = np.expand_dims(mask_array, axis=-1)

    img_array = img_array.astype(np.float32) / 255.0
    mask_array = (mask_array > 0).astype(np.float32)

    X[i] = img_array
    Y[i] = mask_array

# Split dataset
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=2 / 3, random_state=42)


# dice_coef function
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


# -----------------------
# Function to build dynamic U-Net
# -----------------------
def build_unet(input_shape, num_steps, base_filters):
    inputs = tf.keras.layers.Input(input_shape)
    skips = []
    x = inputs
    # Downsampling
    for step in range(num_steps):
        filters = base_filters * (2 ** step)
        x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1 + 0.05 * step)(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            x)
        # x = tf.keras.layers.BatchNormalization()(x)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    filters = base_filters * (2 ** num_steps)
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)

    # Upsampling
    for step in reversed(range(num_steps)):
        filters = base_filters * (2 ** step)
        x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.concatenate([x, skips[step]])
        x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1 + 0.05 * step)(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            x)
        # x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    # Set learning rate to 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[dice_coef])
    return model


# Final CSV
# human-readable timestamp without seconds
timestamp = datetime.datetime.now().strftime("%d%b%Y_%Hh%Mm")
# final CSV name
output_file = f'segmentation_metrics_Skin_{timestamp}.csv'
print(output_file)
if os.path.exists(output_file):
    os.remove(output_file)

# -----------------------
# Grid search combinations loop
# -----------------------
for num_steps in list_num_steps:
    for base_filters in list_base_filters:
        print(f"\n==== Training U-Net: steps={num_steps}, base_filters={base_filters} ====")

        try:
            model = build_unet((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_steps, base_filters)
            
            # Active callbacks only
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',  # monitor loss on the validation set
                    patience=20,         # stop if no improvement for 20 consecutive epochs
                    verbose=1,           # print message when training stops
                    restore_best_weights=True  # restore best weights
                )
            ]

            # Training
            start_time = time.time()
            results = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                batch_size=8,
                epochs=200,
                callbacks=callbacks
            )
            training_time_min = (time.time() - start_time) / 60
            print(f"Training time: {training_time_min:.2f} minutes")

            # Prediction and metrics
            y_pred = model.predict(X_test)
            y_pred_bin = (y_pred > 0.5).astype(np.uint8)
            y_true_bin = Y_test.astype(np.uint8)

            iou_metric = MeanIoU(num_classes=2)
            iou_metric.update_state(y_true_bin, y_pred_bin)
            Mean_IoU = iou_metric.result().numpy()

            y_true_flat = y_true_bin.flatten()
            y_pred_flat = y_pred_bin.flatten()
            cm = confusion_matrix(y_true_flat, y_pred_flat)
            TP = cm[1, 1];
            TN = cm[0, 0];
            FP = cm[0, 1];
            FN = cm[1, 0]
            Sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
            Specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
            Precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            Recall = Sensitivity
            F1 = f1_score(y_true_flat, y_pred_flat)
            Pixel_Accuracy = (TP + TN) / (TP + TN + FP + FN)
            IoU = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
            FW_IoU = sum([(np.sum(y_true_bin == c) / y_true_bin.size) * (
                        np.sum((y_true_bin == c) & (y_pred_bin == c)) / np.sum((y_true_bin == c) | ((y_pred_bin == c)))) for
                          c in range(2)])
            Mean_Accuracy = np.mean([TP / (TP + FN) if (TP + FN) != 0 else 0, TN / (TN + FP) if (TN + FP) != 0 else 0])

            # History + global metric
            history = results.history
            combined_dict = {
                'Num_Steps': num_steps,
                'Base_Filters': base_filters,
                'Sensitivity': Sensitivity,
                'Specificity': Specificity,
                'IoU': IoU,
                'Mean_IoU': Mean_IoU,
                'FW_IoU': FW_IoU,
                'Dice_F1': F1,
                'Pixel_Accuracy': Pixel_Accuracy,
                'Precision': Precision,
                'Recall': Recall,
                'Mean_Accuracy': Mean_Accuracy,
                'Training_Time_min': training_time_min
            }
            for epoch in range(len(history['loss'])):
                e = epoch + 1
                combined_dict[f'Epoch_{e}_Train_Dice'] = history['dice_coef'][epoch]
                combined_dict[f'Epoch_{e}_Val_Dice'] = history['val_dice_coef'][epoch]
                combined_dict[f'Epoch_{e}_Train_Loss'] = history['loss'][epoch]
                combined_dict[f'Epoch_{e}_Val_Loss'] = history['val_loss'][epoch]

            # Write CSV (append)
            df = pd.DataFrame([combined_dict])
            df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            print(f"✅ Metrics saved for steps={num_steps}, base_filters={base_filters}")

        except Exception as e:
            print(f"❌ Error with steps={num_steps}, filters={base_filters}")
            print(e)

            error_dict = {
                'Num_Steps': num_steps,
                'Base_Filters': base_filters,
                'Error': str(e)
            }

            df = pd.DataFrame([error_dict])
            df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

        # --- MEMORY CLEANUP BETWEEN ITERATIONS ---
        finally:
            try:
                del model
            except:
                pass
            K.clear_session()
            gc.collect()