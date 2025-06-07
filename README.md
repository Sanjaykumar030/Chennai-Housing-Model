# 🏠 Chennai Housing Price Prediction using XGBoost 🚀

This project predicts housing prices in Chennai using the XGBoost machine learning algorithm, built entirely in a Jupyter/Google Colab environment. It preprocesses real-world housing data and provides accurate price predictions based on various features such as area, size, amenities, and building quality.

---

## ✨ Features

- ⚡ Built using [XGBoost](https://xgboost.readthedocs.io/en/stable/), a powerful gradient boosting framework  
- 📊 Handles missing values, categorical encoding, and feature scaling  
- 🧠 Predicts housing prices using model trained on Chennai property dataset  
- 📈 Includes model evaluation (MSE, R²) and feature importance visualization  
- 🏡 Predicts the price of a *new custom home input*  

---

## 📁 Project Structure

```
chennai-housing-xgboost/
│
├── main.ipynb         # Jupyter notebook containing all code (Colab compatible)
├── Chennai housing sale.csv  # Dataset used for training
├── requirements.txt   # List of required Python packages
├── README.md          # Project documentation (this file)
└── feature_graph.png      # Importance of each feature for the model's prediction

```

---

## 🛠️ Getting Started

> This project is designed to run easily in **Google Colab** or a local Jupyter environment.

### 🔧 Installation (optional if using Colab)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chennai-housing-xgboost.git
cd chennai-housing-xgboost
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run in Google Colab:
1. Open the notebook `main.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload the dataset when prompted (`Chennai housing sale.csv`).
3. Run all cells.

### Local Jupyter Use:
1. Make sure you have Jupyter installed.
2. Place `main.ipynb` and the CSV dataset in the same folder.
3. Launch Jupyter:
```bash
jupyter notebook
```
4. Open `main.ipynb` and run all cells.
- 📂 Dataset Source: [Chennai Housing Prices](https://www.kaggle.com/datasets/kunwarakash/chennai-housing-sales-price) by Akash Kunwar on Kaggle

---

## 📊 Model Workflow

- 📥 Data Loading (`Chennai housing sale.csv`)
- 🧹 Cleaning (handling dates, nulls, dropping unused columns)
- 🔄 Encoding (one-hot encoding for categorical features)
- 📏 Feature Scaling (StandardScaler)
- 🧠 Model Training (XGBoost Regressor)
- ✅ Evaluation (MSE & R²)
- 🔮 Real-world prediction using manually entered property details

---

## 🖼️ Output Example

```
Mean Squared Error: 48,25,614.23
R-squared: 0.9512

Predicted House Price: ₹62,15,000.00
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements or bug fixes.

🔔 **Note:** Before contributing, please read the Caution section regarding dataset limitations and prediction accuracy.


---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

Feel free to reach out to me:

- Email: [sksanjaykumar010307@gmail.com](mailto:sksanjaykumar010307@gmail)
- LinkedIn: [linkedin.com/in/sanjay-kumar-sakamuri-kamalakar-a67148214](https://linkedin.com/in/sanjay-kumar-sakamuri-kamalakar-a67148214)
- ORCID: [https://orcid.org/0009-0009-1021-2297](https://orcid.org/0009-0009-1021-2297)
## ⚠️ Caution 
- The model is built and trained on a Kaggle dataset last updated in 2022. Therefore, its predictions are based on data available up to that year and may not reflect current market trends.
