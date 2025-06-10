# 📝 IELTS Writing Task 1 Evaluator

A Streamlit app that evaluates IELTS Academic Writing Task 1 essays using Google's Gemini model. Upload a chart/graph or choose a sample, write your essay, and get detailed band scores with actionable feedback based on official IELTS criteria.

---

## 🔍 Features

- 📊 Upload your own IELTS Task 1 image (chart, graph, table)
- ✍️ Paste your written essay in response to the visual
- 🤖 Evaluates using **Google Gemini 1.5 Flash**
- ✅ Scores based on IELTS criteria:
  - Task Achievement
  - Coherence and Cohesion
  - Lexical Resource
  - Grammatical Range and Accuracy
- 📈 Visual band score bar chart
- 💬 Criterion-specific feedback

---

## 🚀 Live App

👉 [Launch the app on Streamlit](https://ielts-task-1-evaluator.streamlit.app/)

---

## 🧠 How It Works

1. Upload or select a visual prompt image.
2. Paste your essay into the editor.
3. The app sends the image and essay to Gemini.
4. Gemini returns:
   - Band scores (0–9)
   - Feedback per criterion
5. Results are displayed as a chart and text.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/gajula21/ielts-task-1-evaluator.git
cd ielts-task-1-evaluator
```

### 2. Add your Google Gemini API key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
ielts-task-1-evaluator/
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .env                  # API key (not committed)
└── README.md             # Project overview
```

---

## 📄 Sample Image Sources

Example images taken from public IELTS resources:

- [IELTS Buddy](https://www.ieltsbuddy.com/)
- [British Council](https://takeielts.britishcouncil.org/)
- [HowToDoIELTS](https://howtodoielts.com/)

---

## 📌 Notes

- Minimum length: 150 words
- Use an objective, formal tone
- Include:
  - Introduction
  - Overview
  - Key details (with data)

---

## 👤 Author

**Vivek Gajula**  
🔗 [github.com/gajula21](https://github.com/gajula21)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---
