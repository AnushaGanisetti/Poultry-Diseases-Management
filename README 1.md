# Transfer Learning Based Classification of Poultry Diseases for Enhanced Health

This project uses transfer learning to classify poultry diseases from chicken images, helping in early detection and better farm health management.

## 🧠 Model
We use a pre-trained convolutional neural network (ResNet50 / MobileNetV2) with fine-tuning for poultry disease classification.

## 📁 Dataset
- Classes: [Healthy, Fowl Pox, Newcastle, Coccidiosis, Avian Influenza]
- Source: [Kaggle or Custom Dataset - specify here]
- Structure:
  ```
  dataset/
  ├── train/
  │   ├── healthy/
  │   ├── fowl_pox/
  │   └── ...
  └── test/
      ├── healthy/
      ├── fowl_pox/
      └── ...
  ```

## 🚀 How to Run

1. Clone this repo:
    ```
    git clone https://github.com/yourusername/poultry-disease-classification.git
    cd poultry-disease-classification
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Train the model:
    ```
    python main.py
    ```

4. Predict on a new image:
    ```
    python predict.py --image path_to_image.jpg
    ```

## 📈 Accuracy
- Training Accuracy: ~95%
- Validation Accuracy: ~90%

## 🤝 Contributions
Feel free to raise issues or pull requests.

## 📄 License
MIT License
