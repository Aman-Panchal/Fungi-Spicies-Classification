# 🍄 Fungi-Spicies-Classification
This project tackles a **fine-grained classification** problem using both **image** and **tabular metadata** to classify **2427 fungi species**, many with only 1–5 samples per class. It leverages a **hybrid deep learning architecture** and extensive preprocessing to maximize accuracy in a highly imbalanced, few-shot setting.
In this we tried EfficientNetB3 and DenseNet121 CNN backbones with metadata for improved classification performance.

---
## 🚀 Project Highlights

- **Model Type**: Hybrid CNN (EfficientNetB3) + Tabular MLP
- **Classes**: 2,427 unique fungi species
- **Data**: Images + Metadata (e.g., habitat, region, elevation, etc.)
- **Augmentation**: Extensive on images using `ImageDataGenerator`
- **Imbalance**: Many classes have only 1–5 samples
- **Optimization**: Focal Loss + Top-5 Accuracy + Mixed Precision
- **Hardware**: Trained with limited resources (15GB GPU, 90 min session)

---
## 📸 Image Processing

- Image size: **380x380**
- Batch size: **2**
- Augmentation includes:
  - Rotation, width/height shift, shear, zoom
  - Horizontal & vertical flip
- Rescaled pixel values to [0, 1]

---
## 🧠 Model Architecture

### Image Branch:
- **Backbone**: EfficientNetB3, DenseNet121 (pretrained on ImageNet)
- **GlobalAveragePooling** + Dense(256) + Dropout

### Tabular Branch:
- MLP with 2 Dense layers: 128 → 256 units
- BatchNormalization + Dropout

### Final Classifier:
- Concatenated both branches → Dense(512) → Softmax
- Output: 2427 class probabilities

---
## 🧮 Loss & Metrics

- **Loss Function**: Focal Loss (gamma=2, alpha=0.25)
- **Metrics**:
  - Categorical Accuracy
  - Top-5 Accuracy (custom metric)

---

## 🧪 Training Strategy

- Initial training with **frozen base model** for 5 epochs
- Fine-tuned entire model for additional 40 epochs
- Custom `ModelCheckpoint` that saves best model every 5 epochs
- Learning rate scheduling with `ReduceLROnPlateau`

---

## ⚙️ Features

- ✅ Mixed Precision Training (`mixed_float16`)
- ✅ Custom data generator to feed image + tabular batches
- ✅ Efficient training loop under GPU and time constraints

---

## 📌 Project Highlights
### ⚙️ Mixed Precision Training
```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
```
- **Why**: Speeds up training and reduces memory usage without loss of model accuracy.

---

## 🧹 Preprocessing & Data Engineering

### 📊 `nan_insertion()`: Smart Imputation
```python
def nan_insertion(dataframe):
    num_cols = dataframe.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        dataframe[col] = dataframe.groupby("category_id")[col].transform(lambda x: x.fillna(x.median()))
    return dataframe
```
- **Why it’s unique**: Imputes NaNs based on class-specific medians to preserve class-based statistical integrity.

---

### 🔄 Image Augmentation with `ImageDataGenerator`
```python
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
```
- **Why**: Useful due to low sample size per class (1–5 images). Prevents overfitting and encourages model generalization.

### 🧾 `flow_from_dataframe()` vs `flow_from_directory()`
- **Used**: `flow_from_dataframe()` because file-label mappings are provided via CSVs—not standard folder structure.

---

## 🔄 Custom Data Generator
```python
def custom_generator(image_gen, tabular_features, filename_to_index):
```
- **Why**: Allows synchronous feeding of tabular and image data per sample.
- **Unique**: Combines the advantages of Keras data generators with external data streams (hybrid models).

---

## 🧠 Custom Callback for Best Model Checkpointing
```python
class SaveBestEvery5Epochs(Callback):
```
- **Why**: Saves best model every 5 epochs based on `val_top_5_accuracy` instead of saving every epoch—efficient and smart.

---

## 🎯 Custom Loss & Metrics

### ⚠️ Focal Loss (For Class Imbalance)
```python
def focal_loss(gamma=2., alpha=0.25):
```
- **Why**: Highly useful for handling extreme class imbalance by down-weighting easy examples.

### 🔝 Top-5 Accuracy Metric
```python
def top_5_accuracy(y_true, y_pred):
```
- **Why**: Helps evaluate near-miss predictions in extreme classification (2427 classes).

---

## 🧠 CNN Models Used

### EfficientNetB3
- **Why**: Good balance of speed and accuracy for fine-grained image classification tasks.

### DenseNet121 *(Optional Alternative)*
- **Why**: Dense connections help preserve information and gradient flow in deep networks.

---

## 🧬 Why Hybrid CNN + Tabular
- **Boosts Accuracy**: Tabular metadata contains useful location, environmental, and observational context.
- **Model learns more** than what’s visible in the image alone.

---

## 🧪 Training Details
- Mixed Precision Enabled ✔️
- Batch Size: 2 (GPU memory optimized)
- Optimizer: Adam (with low `epsilon`)
- Loss: Custom Focal Loss
- Metrics: Accuracy + Top-5 Accuracy
- Callbacks:
  - `ReduceLROnPlateau`
  - `SaveBestEvery5Epochs`

---

## 📈 Performance Monitoring
- Top-5 Accuracy is used to capture performance on fine-grained classes
- Best model is saved during training for deployment or further tuning

---




