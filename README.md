
# Investigating Fibroblast Migration Using Deep Learning

## Overview

This project dives into using deep learning techniques to analyze fibroblast migration during wound healing. Our main focus was to accurately segment wound areas in grayscale time-lapse microscopy images using two AI models — a U-Net and a lightweight CNN — and compare how well they perform.

---

## 🔍 Why This Matters

Fibroblasts play a huge role in wound repair by migrating into the wound space. Traditionally, tracking this migration involves manually segmenting images, which is tedious and time-consuming. With this project, we wanted to automate that process so researchers can quickly quantify wound area changes over time — saving time and improving accuracy.

---

## 🧠 What We Built

We designed a complete image segmentation pipeline using Python and PyTorch. Here’s a breakdown of what’s in the repo:

- Preprocessing script to convert `.tif` stacks into enhanced `.png` images and binary masks  
- Training code for both a **U-Net** and a **basic CNN** model  
- Dice score tracking to compare performance  
- Scripts to visualize the model predictions  
- Model weights and architecture for easy re-use or retraining  

---

## 📊 Key Results

We trained and validated both models on the same dataset and evaluated them using the Dice Score.

| Model     | Final Dice Score (Validation) |
|-----------|-------------------------------|
| **U-Net** | 0.914                         |
| **CNN**   | 0.907                         |

While both models did well, U-Net slightly edged out the CNN in both quantitative accuracy and the visual quality of the segmentation.

---

### 📈 Training Performance

U-Net Dice Score Progress:
![U-Net Training](https://github.com/user-attachments/assets/3cdb5154-4343-4a0e-b5cc-be3bb42c005f)

CNN Dice Score Progress:
![CNN Training](https://github.com/user-attachments/assets/eca3afc9-675e-4275-a8d8-4be28b35fa17)

---

### 🖼 Sample Predictions

U-Net Output:
![U-Net Prediction](https://github.com/user-attachments/assets/179ef56f-fb85-4492-b223-fac041b19b6f)

CNN Output:
![CNN Prediction](https://github.com/user-attachments/assets/a3d30e7e-e365-4031-a6cc-1dc4c6ba9aac)

---

## 📁 What’s Inside This Repo

```
├── preprocess_tiff_stack.py     # Prepares images + binary masks
├── train_unet.py                # Trains the U-Net model
├── train_cnn.py                 # Trains the basic CNN
├── visualize_results.py         # Shows predicted edges overlaid
├── data/
│   ├── images/                  # Processed grayscale images
│   └── masks/                   # Corresponding binary masks
├── requirements.txt             # Python dependencies
```

---

## 🧪 How to Run the Project

### Step 1 — Clone the Repo

```bash
git clone https://github.com/your-username/Investigating-Fibroblast-Migration-Using-Deep-Learning.git
cd Investigating-Fibroblast-Migration-Using-Deep-Learning
```

---

### Step 2 — Preprocess the Data

This converts the raw `.tif` stack into a series of enhanced `.png` images and binary wound masks.

```bash
python preprocess_tiff_stack.py
```

What this script does:
- Converts 16-bit images to 8-bit  
- Applies CLAHE for better contrast  
- Performs Otsu thresholding and cleans the mask  
- Saves processed images to `data/images/` and `data/masks/`

---

### Step 3 — Train the U-Net

```bash
python train_unet.py
```

- Uses BCE + Dice Loss  
- Saves the best model as `small_unet_cpu.pth`  
- Logs Dice scores for each epoch

---

### Step 4 — Train the CNN

```bash
python train_cnn.py
```

- Lighter model, faster to train  
- Good for baseline comparison  
- Also logs Dice score per epoch

---

### Step 5 — Visualize Model Predictions

```bash
python visualize_results.py
```

- Loads a trained model  
- Picks a random validation image  
- Overlays predicted wound edges on the image

---

## 🛠 Requirements

You can install all required packages with:

```bash
pip install -r requirements.txt
```

Main libraries used:
- numpy  
- opencv-python  
- torch  
- tifffile  
- matplotlib  
- scipy  
- tqdm  

---

## 📦 Dataset Info

The dataset we used is currently **private** and can't be published right now since it’s part of an ongoing research paper. However, if you’d like to evaluate the models or try the pipeline, we’re happy to share the dataset for academic use — just reach out via email or GitHub.

---

## 🎥 Project Video

We also recorded a walkthrough of everything we did — from the idea to the model results.

📺 [Click to Watch the Presentation](https://youtu.be/cl-s5VzJd5g)  
[![YouTube Preview](https://img.youtube.com/vi/cl-s5VzJd5g/0.jpg)](https://youtu.be/cl-s5VzJd5g)

---

## 📬 Contact

If you have any questions or want access to the dataset, feel free to reach out:

**Sesha Sai Ramineni**  
📧 ramineniseshasai@gmail.com

---

## 🙌 Acknowledgments

This was done as part of our academic coursework and research on biomedical imaging. Huge thanks to our advisor and everyone who supported us along the way!
