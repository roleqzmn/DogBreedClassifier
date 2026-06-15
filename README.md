## Model Architecture & Training

Two approaches were evaluated for breed classification across 5 breeds (~1,500 images each, 80/10/10 split):

**Baseline: KNN**
- Flattened 128×128 RGB images as raw pixel features
- k=3 neighbors
- **47% test accuracy** — established a baseline but struggled with image variability

**Final Model: VGG16 + Fine-tuning (CNN)**
- Pretrained VGG16 (ImageNet weights) with last 4 layers unfrozen
- Added GlobalAveragePooling → Dense(256) → Dropout(0.5) → Softmax(5)
- Mixed precision training (float16) with GPU memory management
- Data augmentation: rotation, shifts, zoom, brightness, horizontal flip
- Adam (lr=1e-5), early stopping (patience=10)
- **89.2% validation accuracy**

**Breeds:** Border Collie, French Bulldog, Chihuahua, Siberian Husky, Golden Retriever
