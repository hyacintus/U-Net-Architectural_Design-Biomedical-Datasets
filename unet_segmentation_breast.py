import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import img_to_array, load_img
from skimage.transform import resize
from tqdm import tqdm
import cv2
import time
import gc
from tensorflow.keras import backend as K
import datetime
import random

# -----------------------
# Dataset paths
# -----------------------
image_dir = r"C:\Users\PATH\Images"
mask_dir = r"C:\Users\PATH\Masks"

# -----------------------
# Grid search combinations list
# -----------------------
list_num_steps = [5]  # number of U-Net steps   [3, 4, 5]
list_base_filters = [64]  # starting filters   [4, 8, 16, 32, 64]

# -----------------------
# File list
# -----------------------
image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])

# -----------------------
# Random reproducible subset selection
# -----------------------
MAX_IMAGES = 3032  # number of images to use: 3032, but max is probably 1000
random.seed(42)  # fixed seed for reproducibility

combined = list(zip(image_filenames, mask_filenames))
random.shuffle(combined)

image_filenames, mask_filenames = zip(*combined)
image_filenames = list(image_filenames[:MAX_IMAGES])
mask_filenames = list(mask_filenames[:MAX_IMAGES])

IMG_WIDTH = 1024  # Not used, only kept as a reminder of the image dimensions
IMG_HEIGHT = 1024  # Not used, only kept as a reminder of the image dimensions
PATCH_SIZE = 128
STRIDE = PATCH_SIZE // 2
IMG_CHANNELS = 1

# -----------------------
# Image split
# -----------------------
train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
    image_filenames, mask_filenames, test_size=0.3, random_state=42
)
val_imgs, test_imgs, val_masks, test_masks = train_test_split(
    temp_imgs, temp_masks, test_size=2 / 3, random_state=42
)


# -----------------------
# Patch + preprocess functions
# -----------------------
def create_patches_sliding(img, mask, patch_size=PATCH_SIZE, stride=STRIDE):
    patches_img, patches_mask = [], []
    H, W = img.shape[:2]
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_patch = img[y:y + patch_size, x:x + patch_size]
            mask_patch = mask[y:y + patch_size, x:x + patch_size]
            lesion_ratio = np.sum(mask_patch) / (patch_size ** 2)
            if lesion_ratio > 0.05 or random.random() < 0.015:
            # if (lesion_ratio > 0.05 and random.random() < 0.25) or \
            #         (lesion_ratio <= 0.05 and random.random() < 0.00375):
                patches_img.append(img_patch)
                patches_mask.append(mask_patch)
    return patches_img, patches_mask


def process_dataset(image_list, mask_list):
    X, Y = [], []
    for img_name, mask_name in zip(image_list, mask_list):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_array = np.expand_dims(mask_array, axis=-1)

        if img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)

        # Normalization and CLAHE
        img_array = img_array.squeeze()
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_array = clahe.apply(img_array)
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = img_array.astype(np.float32) / 255.0

        mask_array = (mask_array > 0).astype(np.float32)

        img_patches, mask_patches = create_patches_sliding(img_array, mask_array)
        X.extend(img_patches)
        Y.extend(mask_patches)

        # 🔥 CLEANUP
        del img_array, mask_array, img_patches, mask_patches
        gc.collect()

    return np.array(X), np.array(Y)


def print_patch_stats(Y, name="DATASET"):
    lesion_ratios = [np.mean(patch) for patch in Y]

    high = sum(r > 0.05 for r in lesion_ratios)
    low = sum(r <= 0.05 for r in lesion_ratios)
    total = len(lesion_ratios)

    print(f"\n📊 STATISTICS {name}")
    print(f"Total patches: {total}")
    print(f"Patches >5% lesion: {high} ({100 * high / total:.2f}%)")
    print(f"Patches <=5% lesion: {low} ({100 * low / total:.2f}%)")


print("\nCreating TRAIN patches...")
X_train, Y_train = process_dataset(train_imgs, train_masks)
print_patch_stats(Y_train, "TRAIN")

print("\nCreating VAL patches...")
X_val, Y_val = process_dataset(val_imgs, val_masks)
print_patch_stats(Y_val, "VALIDATION")

print("\nCreating TEST patches...")
X_test, Y_test = process_dataset(test_imgs, test_masks)
print_patch_stats(Y_test, "TEST")


# -----------------------
# Dice function
# -----------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


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
    for step in range(num_steps):
        filters = base_filters * (2 ** step)
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.1 + 0.05 * step)(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    filters = base_filters * (2 ** num_steps)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Upsampling
    for step in reversed(range(num_steps)):
        filters = base_filters * (2 ** step)
        x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.concatenate([x, skips[step]])
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.1 + 0.05 * step)(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_coef])
    return model


# Final CSV
# human-readable timestamp without seconds
timestamp = datetime.datetime.now().strftime("%d%b%Y_%Hh%Mm")
# final CSV name
output_file = f'segmentation_metrics_Breast_{timestamp}.csv'
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
            model = build_unet((PATCH_SIZE, PATCH_SIZE, IMG_CHANNELS), num_steps, base_filters)

            # Active callbacks only
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',  # monitor loss on the validation set
                    patience=50,         # stop if no improvement for 50 consecutive epochs
                    verbose=1,           # print message when training stops
                    restore_best_weights=True  # restore best weights
                )
            ]

            start_time = time.time()
            results = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                batch_size=16,
                epochs=200,
                callbacks=callbacks
            )
            training_time_min = (time.time() - start_time) / 60
            print(f"Training time: {training_time_min:.2f} minutes")

            y_pred = model.predict(X_test)
            y_pred_bin = (y_pred > 0.5).astype(np.uint8)
            y_true_bin = (Y_test > 0).astype(np.uint8)

            cm = confusion_matrix(y_true_bin.flatten(), y_pred_bin.flatten(), labels=[0, 1])
            TN, FP, FN, TP = cm.ravel()
            Sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
            Specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
            Precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            Recall = Sensitivity
            F1 = f1_score(y_true_bin.flatten(), y_pred_bin.flatten(), zero_division=0)
            Pixel_Accuracy = (TP + TN) / (TP + TN + FP + FN)
            IoU = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
            num_classes = 2
            mean_iou_metric = MeanIoU(num_classes=num_classes)
            mean_iou_metric.update_state(y_true_bin, y_pred_bin)
            Mean_IoU = mean_iou_metric.result().numpy()
            pixel_count_class = [np.sum(y_true_bin == c) for c in range(num_classes)]
            total_pixels = y_true_bin.size
            IoU_per_class = [
                np.sum((y_true_bin == c) & (y_pred_bin == c)) / np.sum((y_true_bin == c) | (y_pred_bin == c)) if np.sum(
                    (y_true_bin == c) | (y_pred_bin == c)) > 0 else 0 for c in range(num_classes)]
            FW_IoU = sum((pixel_count_class[c] / total_pixels) * IoU_per_class[c] for c in range(num_classes))
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

            df = pd.DataFrame([combined_dict])
            df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            print(f"✅ Metrics saved for steps={num_steps}, base_filters={base_filters}")

        except Exception as e:
            print(f"❌ Error with steps={num_steps}, filters={base_filters}")
            print(e)
            error_dict = {
                'Num_Steps': num_steps,
                'Base_Filters': base_filters,
                'Error': str(e),
                'Status': 'FAILED'
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