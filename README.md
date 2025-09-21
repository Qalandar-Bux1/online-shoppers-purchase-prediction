🛒 Online Shoppers Purchase Intention Prediction

This project predicts whether an online shopper will make a purchase based on their browsing behavior using Machine Learning (ML) and Deep Learning (ANN) techniques.

The goal is to compare classical ML models with a neural network to determine which approach provides better accuracy and insights for e-commerce businesses.




📊 Dataset

Name: Online Shoppers Purchase Intention Dataset

Size: ~12,000 rows, 18 features

Source: UCI Machine Learning Repository

Description: Includes session-level features like:

Number of pages visited

Time spent on different page types



Revenue (target variable)




Logistic Regression

Random Forest

Deep Learning:

Artificial Neural Network (ANN) built with TensorFlow/Keras

📈 Evaluation Metrics

The models were evaluated using:

For Classification:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Score

Visualizations:

Confusion Matrix

Loss & Accuracy Plots (for ANN)

Model performance comparison table

📝 Results Summary
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.86	0.83	0.79	0.81	0.88
Random Forest	0.90	0.88	0.85	0.86	0.92
ANN (Deep Learning)	0.93	0.91	0.90	0.90	0.95

Key Insight:

The ANN outperformed both classical models, achieving the highest accuracy and recall.
Random Forest was a strong alternative, especially for interpretable and faster predictions.

🚀 Key Learnings

Data preprocessing and feature engineering are critical for performance.

Deep Learning excels in complex data patterns but requires more compute time.

Random Forest provides a balance between accuracy and interpretability.

Logistic Regression serves as a good baseline model.

💡 Future Improvements

Hyperparameter tuning for ANN and Random Forest.

Integration with real-time e-commerce data streams.

Deploying the final model as a web app using Flask or FastAPI.

Exploring advanced deep learning architectures like RNNs or Transformers.

🖼️ Visuals
Confusion Matrix Example:

ANN Training Performance:

🧰 Tech Stack

Programming Language: Python

Libraries:

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

TensorFlow / Keras

Jupyter Notebook
