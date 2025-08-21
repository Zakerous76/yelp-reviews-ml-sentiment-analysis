
# Opinion Mining on Web Reviews  
**Web Mining Project**  

---

## 1. Introduction
In the era of digital commerce and online interaction, platforms like Yelp host vast volumes of user-generated content that reflect customer experiences. These reviews are valuable for businesses seeking to understand public perception.  

However, the scale of data makes manual analysis impractical. This project tackles this issue through **opinion mining**, a subfield of **Natural Language Processing (NLP)**, to automatically extract and classify sentiments from textual reviews. The goal is to create an intelligent system that classifies reviews as **positive, neutral, or negative**, and presents predictions through a user-friendly web interface.  

---

## 2. Problem Statement
Given the sheer volume and diverse nature of online reviews, manual sentiment analysis is both inefficient and prone to subjectivity. Automating this process through machine learning offers a scalable, consistent, and cost-effective alternative.  

This project explores and compares multiple machine learning models to determine the most effective approach for sentiment classification on real-world review data. Additionally, a **Streamlit-based web interface** was developed to allow users to interactively test the implemented models.  

---

## 3. Objectives
The primary objective is to build an automated sentiment analysis system capable of classifying user reviews into sentiment categories. Specific objectives include:  

- Analyze and preprocess Yelp review data for different experimental approaches.  
- Conduct initial experiments using **Logistic Regression** as a baseline.  
- Apply three classification models:  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Random Forest (and Naïve Bayes)  
- Evaluate model performance using **accuracy, precision, recall, and F1-score**.  
- Perform hyperparameter tuning on the best-performing model.  
- Develop a **user-friendly Streamlit web interface** for real-time sentiment prediction.  

---

## 4. Dataset Description

### 4.1 Yelp Review Dataset
This project utilizes the **Yelp Review Dataset** (~8.65 GB in JSON format), containing:  

- `business.json` – Business information, locations, attributes, categories.  
- `review.json` – Full review texts, user_id, business_id, star ratings.  
- `user.json` – User metadata, social connections, review history.  
- `checkin.json` – Records of user check-ins.  
- `tip.json` – Short user tips.  
- `photo.json` – User-uploaded photo metadata.  

#### 4.1.1 Focused Dataset
The project focuses on `review.json` for sentiment classification.  

Class distribution before sampling:  
```

1★: 1,069,561 | 2★: 544,240 | 3★: 691,934 | 4★: 1,452,918 | 5★: 3,231,627

````
Balanced sampling was performed to have **100,000 reviews per rating**, creating datasets of up to **600,000 entries** for training and evaluation.  

---

## 5. Methodology

### 5.1 Preprocessing
Reviews were labeled based on star ratings:  
- 1★ → Negative  
- 4-5★ → Positive  
- Neutral labels were experimented with using 2★ and 3★ reviews.  

Text preprocessing like stopword removal or normalization was skipped because **TF-IDF vectorization** was used, preserving emotional nuances in reviews.

**Labeling experiments:**  

| Experiment | Negative | Neutral | Positive |
|-----------|----------|---------|---------|
| A         | 1★       | 2★      | 4★/5★   |
| B         | 1★       | 2★+3★   | 4★/5★   |
| C         | 1★       | 3★      | 4★/5★   |
| D         | 1★       | None    | 4★/5★   |

### 5.2 Model Training
Three approaches were implemented:  

1. **Ternary Classification (3-class)** – Predicts positive, neutral, negative. Weighted F1-score ~80%.  
2. **Binary Classification (2-class)** – Predicts positive vs. negative. Accuracy ~96.6%.  
3. **Multiclass Classification (5-class)** – Predicts exact star rating. Accuracy ~60.1%.  

All models were trained on **class-balanced datasets** with a 50/50 train-test split.  

---

## 6. Results
- **Approach One (Ternary)** – Weighted F1: 82.0%, Neutral class hardest to predict.  
- **Approach Two (Binary)** – Accuracy: 96.6%, F1-score: 96.6%, best overall.  
- **Approach Three (Multiclass)** – Accuracy: 60.1%, Logistic Regression performed best.  

> Confusion matrices showed misclassification mostly between adjacent classes, highlighting subjective sentiment in borderline reviews.  

---

## 7. Web Interface
A **Streamlit app** was built to demonstrate sentiment classification. Features include:  

- Choose a model (Binary, Ternary, Star Rating).  
- Enter review text and get real-time predictions.  
- Visual representation of probabilities or predicted sentiment using colors and emojis.  

**Run the app:**  
```bash
streamlit run streamlit_app.py
```


## 8. Conclusion & Future Work

The project demonstrates effective sentiment mining from Yelp reviews.

* **Binary classification** provided the highest accuracy.
* **Ternary classification** offered balanced insights.
* **Streamlit interface** allows interactive user experience.

**Future work:**

* Explore deep learning models like **BERT**.
* Integrate business metadata for richer analysis.
* Deploy as a full web service using **FastAPI and Docker**.

---

## 9. References

1. [Yelp Open Dataset](https://www.yelp.com/dataset)

---

## 10. Performance 

<img width="1325" height="804" alt="image" src="https://github.com/user-attachments/assets/eea444c1-00aa-4b27-8b57-27cf6078c1b9" />

