# 💳 Bank Customer Segmentation using Clustering (Unsupervised ML)

This Flask web app segments bank customers into meaningful groups using multiple unsupervised clustering techniques:  
**HDBSCAN, KMeans, DBSCAN, Agglomerative Clustering**

It automatically:
- Selects the best model using **silhouette score**
- Visualizes the results
- Explains why that model performed best

---

## 🚀 Features

- 📂 Upload a CSV with customer features
- 🧠 Automatically runs 4 clustering models
- 🏆 Picks the best model using silhouette score
- 📊 Displays result, chart, and explanation
- 🔍 Uses PCA + Autoencoder + UMAP for better clustering

---

## 🧪 Installation

```bash
git clone https://github.com/7206582513/UpdatedUnsupervisedModel.git
cd UpdatedUnsupervisedModel

python -m venv venv
venv\Scripts\activate         # (For Windows) or source venv/bin/activate (for Linux/Mac)

pip install --upgrade pip
pip install -r requirements.txt

python app.py
Then open http://127.0.0.1:5000 in your browser.

📁 Project Structure
.
├── app.py
├── utils.py
├── requirements.txt
├── .gitignore
├── README.md
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── cluster_plot.png (auto-generated)

🛠 Tech Stack
Python 3.12

Flask (Web Framework)

Scikit-learn, HDBSCAN, UMAP-learn

TensorFlow (Autoencoder)

Matplotlib + Seaborn (for plotting)

🧠 Author
Rohit Makani
B.Tech Data Science & AI — IIT Guwahati
GitHub: 7206582513

📌 To-Do (Future Enhancements)
 PDF Report export with all charts

 Add interpretability (e.g., SHAP for cluster meaning)

 Add auto-summary of each customer segment

