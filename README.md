# Fitness-Tracker-FitAura-

**FitAura** is a modern fitness tracking and recommendation web application powered by machine learning and natural language processing. It helps users track weight progress over time and get quick answers to fitness-related queries using an FAQ chatbot.

---

## 📌 Key Features

- 📈 **Weight Prediction:** Predicts future weight trends using Support Vector Regression (SVR).
- 💬 **AI Chatbot:** Answers common fitness questions from a curated FAQ dataset.
- 📊 **Data Visualization:** Interactive graphs for weight tracking.
- 📅 **Personal Progress Monitoring:** Users can log and visualize daily weight logs.
- 🧠 **Smart Query Matching:** Uses cosine similarity and TF-IDF to fetch accurate FAQ responses.

---

## ⚙️ Technologies Used

- **Backend:** Python, Flask
- **Machine Learning:** `sklearn.svm.SVR` for regression modeling
- **NLP:** `TfidfVectorizer`, `cosine_similarity`
- **Frontend:** HTML, CSS, Bootstrap
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Plotly

---

## 🧠 AI Model Used

### Support Vector Regression (SVR)

- **Kernel:** RBF (Radial Basis Function)
- **C:** 1000
- **Gamma:** 0.01
- **Epsilon:** 0.1

This configuration enables smooth prediction of future weight trends based on historical data.

---

## 🔄 Data Preprocessing

### 1. FAQ Dataset
- Load dataset from CSV.
- Validate presence of `Question` and `Answer` columns.
- Vectorize text using `TfidfVectorizer`.

### 2. Weight Logs
- Convert list to DataFrame.
- Convert `date` column to datetime.
- Create `days` column for number of days since first log.

### 3. User Query
- Vectorize input query using same vectorizer.
- Compute similarity against all FAQ vectors.
- Retrieve closest matching answer.

---

## 🧪 Training & Testing Logic

- All data from `weight_logs` is used to train the SVR model.
- Predictions are made for future dates without a traditional train/test split.
- Code for training lives in `/generate_graph` and `/predict_future/<int:days>` routes.

Example snippet:
```python
svr_model.fit(X, y)
future_X = np.array([[df['days'].max() + i] for i in range(1, future_days + 1)])
predictions = svr_model.predict(future_X).tolist()


📁 Folder Structure

FitAura-TheFitnessSolution
│
├── app.py                     # Main Flask application
├── utils.py                   # Core logic for weight prediction and FAQ matching
├── templates/
│   ├── index.html             # Homepage UI
│   └── result.html            # Results display page
├── static/                    # Static CSS and JS files
├── conversational_dataset.csv # FAQ data
├── weight_logs.json           # Sample weight data (if any)
├── README.md                  # Project documentation


🎯 How It Works

1. User logs weight entries by date.
2. The model trains on these logs and forecasts future weights.
3. The user can also enter fitness queries in natural language.
4. The chatbot returns best-matched answers from a dataset of FAQs.


📚 References
Scikit-learn Documentation: https://scikit-learn.org/

Flask Documentation: https://flask.palletsprojects.com/
