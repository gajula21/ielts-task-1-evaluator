import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import LLMChain
from PIL import Image
from io import BytesIO
import requests
import base64
import google.generativeai as genai
import random
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="IELTS Task 1 Evaluator",
    page_icon="📝",
    layout="wide"
)

st.title("IELTS Writing Task 1 Evaluator")
st.markdown("""
This application evaluates IELTS Writing Task 1 essays based on official IELTS criteria,
including visual prompts (charts, graphs, tables, images).
Upload your own visual or use the example to get started.
""")

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it as an environment variable (e.g., in a .env file).")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

example_data = [
    {"image_url": "https://howtodoielts.com/wp-content/uploads/2025/03/Screenshot-2025-03-04-181123-compressed.jpg"},
    {"image_url": "https://howtodoielts.com/wp-content/uploads/2025/01/Screenshot-2025-01-28-071405-compressed.jpg"},
    {"image_url": "https://www.ieltsbuddy.com/images/ielts-task-1-pie-chart.jpg"},
    {"image_url": "https://www.ieltsbuddy.com/images/ielts-task-1-bar-chart-UK-G.jpg"},
    {"image_url": "https://www.ieltsbuddy.com/images/ielts-pie-chart-electricity.jpg"},
    {"image_url": "https://www.ieltsbuddy.com/images/ielts-line-graph-cars.jpg"},
    {"image_url": "https://www.ieltsbuddy.com/images/ielts-bar-chart-monthly-expenditure.jpg"},
    {"image_url": "https://www.ieltsbuddy.com/images/ielts-line-graph-books-borrowed-libraries.jpg"},
    {"image_url": "https://www.ieltsbuddy.com/images/task-1-table-sample.png"},
    {"image_url": "https://takeielts.britishcouncil.org/sites/default/files/styles/bc-landscape-800x450/public/ac_writing_task_1_-_2.png?itok=jOVoH4vT"}
]

if "essay_text" not in st.session_state:
    st.session_state.essay_text = ""
if "image" not in st.session_state:
    st.session_state.image = None
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "results" not in st.session_state:
    st.session_state.results = {}

def load_example():
    selected_example = random.choice(example_data)
    st.session_state.essay_text = ""
    example_image_url = selected_example["image_url"]

    try:
        headers = {"User-Agent": "IELTS-Task1-Evaluator/1.0 (https://yourdomain.com/)"}
        res = requests.get(example_image_url, headers=headers, timeout=10)
        res.raise_for_status()
        st.session_state.image_bytes = res.content
        st.session_state.image = Image.open(BytesIO(st.session_state.image_bytes))
        st.success("Example image loaded successfully. Now write your essay based on this visual.")
    except Exception as e:
        st.error(f"Failed to load example image: {e}")
        st.session_state.image_bytes = None
        st.session_state.image = None

with st.sidebar:
    st.button("Load Example", on_click=load_example)
    st.markdown("## Upload Your Visual")
    uploaded_file = st.file_uploader("Upload a visual prompt image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.session_state.image = image
            uploaded_file.seek(0)
            st.session_state.image_bytes = uploaded_file.read()
            st.session_state.essay_text = ""
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            st.session_state.image = None
            st.session_state.image_bytes = None

essay_value = st.session_state.get("essay_text", "")
image_value = st.session_state.get("image", None)

if image_value:
    st.markdown("### Visual Prompt")
    st.image(image_value, caption="Loaded Visual Prompt", use_container_width=True)
else:
    st.info("Upload an image or click 'Load Example' in the sidebar to get started.")

with st.form("task1_form"):
    essay_text = st.text_area(
        "Enter Your Task 1 Essay (Describe the visual above)",
        value=essay_value,
        placeholder="Write your essay describing the visual prompt (chart, graph, table, or diagram) here...",
        height=300,
    )
    submitted = st.form_submit_button("Evaluate Essay")

def evaluate_task1(essay_text, image_bytes):
    try:
        model_name = "gemini-2.0-flash"
        st.info(f"Using hardcoded model: `{model_name}`.")

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0
        )

        task_response = ResponseSchema(name="task_response", description="An object with two keys: 'score' (a numerical string, e.g. '3.0') and 'comment'.")
        coherence_and_cohesion = ResponseSchema(name="coherence_and_cohesion", description="An object with two keys: 'score' and 'comment'.")
        lexical_resource = ResponseSchema(name="lexical_resource", description="An object with two keys: 'score' and 'comment'.")
        grammatical_range_and_accuracy = ResponseSchema(name="grammatical_range_and_accuracy", description="An object with two keys: 'score' and 'comment'.")

        schemas = [task_response, coherence_and_cohesion, lexical_resource, grammatical_range_and_accuracy]
        structured_output_parser = StructuredOutputParser.from_response_schemas(schemas)
        format_instructions = structured_output_parser.get_format_instructions()

        image_part = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
            }
        }

        prompt_content_parts = [
            {
                "type": "text",
                "text": (
                    "You are an IELTS Writing Task 1 evaluator. The user has provided an image "
                    "which represents the visual prompt for a Task 1 essay. "
                    "Below is the candidate's essay describing this visual information. "
                    "Your task is to evaluate the essay based on how well it describes the provided image and "
                    "adheres to official IELTS Writing Task 1 criteria.\n\n"
                    "IELTS Task 1 essays require candidates to summarize, describe, or explain the information "
                    "presented in a visual (graph, table, chart, map, or diagram). They should identify main features, "
                    "report specific details, and make relevant comparisons without offering opinions or conclusions. "
                    "A minimum word count of 150 words is expected.\n\n"
                )
            },
            image_part,
            {
                "type": "text",
                "text": f"Candidate's Essay:\n{essay_text}\n\n"
            },
            {
                "type": "text",
                "text": (
                    "Evaluate on these criteria:\n"
                    "1. task_response\n"
                    "2. coherence_and_cohesion\n"
                    "3. lexical_resource\n"
                    "4. grammatical_range_and_accuracy\n\n"
                    "For each criterion, provide a score and a detailed comment. "
                    f"{format_instructions}\n\n"
                    "Do not include any additional text or explanation or overall band score outside the JSON."
                )
            }
        ]

        messages = [HumanMessage(content=prompt_content_parts)]
        response = llm.invoke(messages)
        result_text = response.content
        structured_result = structured_output_parser.parse(result_text)

        try:
            scores = []
            for criterion in ['task_response', 'coherence_and_cohesion', 'lexical_resource', 'grammatical_range_and_accuracy']:
                if criterion in structured_result and isinstance(structured_result[criterion], dict) and 'score' in structured_result[criterion]:
                    try:
                        scores.append(float(structured_result[criterion]['score']))
                    except ValueError:
                        pass

            if scores:
                average_score = sum(scores) / len(scores)
                remainder = average_score - int(average_score)

                if remainder >= 0.75:
                    final_band_score = int(average_score) + 1.0
                elif remainder >= 0.25:
                    final_band_score = int(average_score) + 0.5
                else:
                    final_band_score = float(int(average_score))

                structured_result['band_score'] = final_band_score
            else:
                structured_result['band_score'] = 'N/A'
        except Exception as e:
            st.warning(f"Error during band score rounding: {e}.")
            structured_result['band_score'] = 'N/A'

        return structured_result

    except Exception as e:
        st.error(f"An unexpected error occurred during evaluation: {e}")
        return {"error": str(e)}

if submitted:
    if not essay_text.strip():
        st.error("Please enter your essay.")
    elif not st.session_state.image_bytes:
        st.error("Please upload or load a visual prompt image for evaluation.")
    else:
        with st.spinner("Evaluating your essay with visual prompt..."):
            results = evaluate_task1(essay_text, st.session_state.image_bytes)
            st.session_state.results = results
        if "error" not in st.session_state.results:
            st.success("Evaluation complete!")

def parse_metric(metric):
    if isinstance(metric, str):
        try:
            return json.loads(metric)
        except json.JSONDecodeError:
            return {"score": "N/A", "comment": metric}
    return metric

if st.session_state.results and "error" not in st.session_state.results:
    results = st.session_state.results
    for key in ['task_response', 'coherence_and_cohesion', 'lexical_resource', 'grammatical_range_and_accuracy']:
        if key in results:
            results[key] = parse_metric(results[key])

    st.markdown("### Overall Evaluation")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 10px; border: 1px solid #ddd; text-align: center; background-color: #262730;">
                <h1 style="font-size: 48px; color: #1e88e5;">{results.get('band_score', 'N/A')}</h1>
                <p style="font-size: 18px;">Overall Band Score</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Detailed Criteria Analysis")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Task Response",
        "Coherence & Cohesion",
        "Lexical Resource",
        "Grammatical Range & Accuracy"
    ])

    with tab1:
        tr = results.get('task_response', {'score': 'N/A', 'comment': 'No evaluation provided.'})
        st.markdown(f"**Score:** {tr['score']}")
        st.markdown(f"**Comments:** {tr['comment']}")

    with tab2:
        cc = results.get('coherence_and_cohesion', {'score': 'N/A', 'comment': 'No evaluation provided.'})
        st.markdown(f"**Score:** {cc['score']}")
        st.markdown(f"**Comments:** {cc['comment']}")

    with tab3:
        lr = results.get('lexical_resource', {'score': 'N/A', 'comment': 'No evaluation provided.'})
        st.markdown(f"**Score:** {lr['score']}")
        st.markdown(f"**Comments:** {lr['comment']}")

    with tab4:
        gr = results.get('grammatical_range_and_accuracy', {'score': 'N/A', 'comment': 'No evaluation provided.'})
        st.markdown(f"**Score:** {gr['score']}")
        st.markdown(f"**Comments:** {gr['comment']}")

    st.markdown("### Score Visualization")

    categories = ['Task Response', 'Coherence & Cohesion', 'Lexical Resource', 'Grammar']
    scores = []
    try:
        scores.append(float(results.get('task_response', {}).get('score', 0)))
        scores.append(float(results.get('coherence_and_cohesion', {}).get('score', 0)))
        scores.append(float(results.get('lexical_resource', {}).get('score', 0)))
        scores.append(float(results.get('grammatical_range_and_accuracy', {}).get('score', 0)))
    except ValueError:
        st.warning("Could not parse all scores for visualization.")
        scores = [s if isinstance(s, (int, float)) else 0 for s in scores]

    plt.style.use('dark_background')
    sns.set_style('dark')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#262730')
    ax.set_facecolor('#262730')
    bars = ax.bar(categories, scores, color=['#5d76cb', '#546ab7', '#4a5ea2', '#41538e'])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}', ha='center', va='bottom', color='white')

    ax.set_ylim(0, 10)
    ax.set_ylabel('Score', color='white', fontweight='bold')
    ax.set_title('IELTS Writing Task 1 Scores by Category', color='white', fontweight='bold')

    overall_band_score = results.get('band_score')
    if overall_band_score is not None and isinstance(overall_band_score, (int, float)):
        try:
            ax.axhline(y=float(overall_band_score), color='red', linestyle='--', label=f"Overall: {overall_band_score}")
        except ValueError:
            pass

    ax.tick_params(axis='x', rotation=0, labelcolor='white')
    ax.tick_params(axis='y', labelcolor='white')
    ax.legend()
    st.pyplot(fig)

with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This application uses AI to evaluate IELTS Writing Task 1 essays based on official IELTS marking criteria:

    * **Task Response**
    * **Coherence and Cohesion**
    * **Lexical Resource**
    * **Grammatical Range and Accuracy**

    The evaluation provides scores on a scale of 1-9 for each criterion, as well as an overall band score.
    """)
    st.markdown("## Tips for IELTS Writing")
    st.markdown("""
    * **Study the visual carefully.** Understand what information it presents and what the main trends or features are.
    * **Identify main features and trends.** Do not describe every single detail.
    * **Organize your essay logically.** Use an introduction, overview of main features, detailed body paragraphs, and a brief conclusion.
    * **Support your descriptions with data.** Reference specific numbers, percentages, or categories from the visual.
    * **Use a variety of vocabulary and grammar.** Avoid repetition.
    * **Aim for at least 150 words.**
    * **Proofread your work** for any errors in grammar, spelling, or punctuation.
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>IELTS Writing Task 1 Evaluator</p>
</div>
""", unsafe_allow_html=True)
