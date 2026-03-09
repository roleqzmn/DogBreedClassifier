import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize
import joblib


selected_breeds = [
    'n02106166-Border_collie',
    'n02108915-French_bulldog',
    'n02085620-Chihuahua',       # Chihuahua
    'n02110185-Siberian_husky',  # Husky
    'n02099601-golden_retriever' # Golden Retriever
]
def extract_features(image_path, target_size=(128, 128)):
    img = imread(image_path)
    img = resize(img, target_size, anti_aliasing=True)
    return img.flatten()
X = []
y = []

for breed in selected_breeds:
    breed_dir = f'C:/Users/wujci/Documents/FITproject/dogs/{breed}'
    for img_file in os.listdir(breed_dir)[:1200]:
        img_path = os.path.join(breed_dir, img_file)
        try:
            features = extract_features(img_path)
            X.append(features)
            y.append(breed.split('-')[1])
        except:
            continue

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
joblib.dump(model, 'dog_breed_knn.pkl')