Accent Classification using Variational Autoencoder (VAE)

Project Overview:
This project implements an accent classification system using MFCC speech features. A Variational Autoencoder (VAE) is used to generate synthetic data in order to improve model performance when the available dataset is small.

Dataset
* Total samples: 329
* Features: 12 MFCC coefficients
* Accent classes: ES, FR, GE, IT, UK, US

Methodology
* Extract MFCC features from speech data
* Train a Variational Autoencoder for data augmentation
* Generate synthetic MFCC samples
* Train and evaluate classifiers on original and augmented data

VAE Architecture:
* Encoder: 12 → 8 → 2
* Decoder: 2 → 8 → 12

Models Used
* Support Vector Machine (SVM)
* Random Forest
* k-Nearest Neighbors (k-NN)

Results
* Best baseline accuracy: 80.30%
* Best accuracy with VAE augmentation: 81.82%

VAE-generated data improves classification performance compared to training on original data alone.


Technologies
* Python
* TensorFlow / Keras
* Scikit-learn
* Librosa
* NumPy


Author

Divya Shree
GitHub: [https://github.com/Divyashree-Kumar](https://github.com/Divyashree-Kumar)

If you want, I can make an **even shorter version (resume-ready)** or **format it for college project submission**.
