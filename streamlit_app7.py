import os
import re
import streamlit as st
import pandas as pd
import spacy
from spacy import displacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
import ast
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['stopwords','wordnet'])

import subprocess

# @st.cache_resource
# def download_en_core_web_sm():
#     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# download_en_core_web_sm()

custom_stopwords = ["city", "state"]

# Function to load the OpenAI API key
def load_openai_api_key():
    # Check if the API key exists in Streamlit secrets
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        return st.secrets["openai"]["api_key"]
    # Fallback to environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    # Raise an error if the key is not found
    raise ValueError("OpenAI API key is not set. Add it to Streamlit secrets or as an environment variable.")


# Load the API key
openai.api_key = load_openai_api_key()

# # Load SpaCy model
# try:
#     # Try to load the model
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     # Download the model if not available
#     spacy.cli.download("en_core_web_sm")
#     nlp = spacy.load("en_core_web_sm")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load job postings with precomputed embeddings
@st.cache_data
def load_data():
    jobs = pd.read_csv("job_postings_with_embeddings_openai.csv")
    resumes = pd.read_csv("resumes_with_updated_summaries.csv")  # Load corpus of resumes
    jobs['description_embedding'] = jobs['description_embedding'].apply(ast.literal_eval).apply(np.array)
    return jobs, resumes

# Generate OpenAI embeddings for the input resume
def generate_openai_embedding(text, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(input=text, model=model)
        return np.array(response['data'][0]['embedding'])  # Convert to numpy array
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# Highlight skills and entities using spaCy's displacy
def highlight_skills_with_displacy(text):
    doc = nlp(text)
    colors = {
        "Job-Category": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
        "SKILL": "linear-gradient(90deg, #9BE15D, #00E3AE)",
        "ORG": "#ffd966",
        "PERSON": "#e06666",
        "GPE": "#9fc5e8",
        "DATE": "#c27ba0",
        "ORDINAL": "#674ea7",
        "PRODUCT": "#f9cb9c",
    }
    options = {
        "ents": [
            "Job-Category",
            "SKILL",
            "ORG",
            "PERSON",
            "GPE",
            "DATE",
            "ORDINAL",
            "PRODUCT",
        ],
        "colors": colors,
    }
    html = displacy.render(doc, style="ent", options=options)
    return html

# Compute similarity and find top job matches
def find_top_matches(resume_embedding, job_embeddings, job_titles, job_descriptions, job_urls, top_n=5):
    # Ensure resume embedding is a 2D array
    resume_embedding = resume_embedding.reshape(1, -1)  # Reshape to (1, -1)
    similarities = cosine_similarity(resume_embedding, job_embeddings)[0]  # Calculate similarities
    top_indices = similarities.argsort()[-top_n:][::-1]  # Get indices of top N matches
    return [(job_titles[i], job_descriptions[i], job_urls[i], similarities[i]) for i in top_indices]

# Extract information for summarization using SpaCy
def extract_resume_info(resume_text):
    """
    Extract key information from a resume using SpaCy.
    """
    doc = nlp(resume_text)
    
    # Extract skills and key entities
    skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "ABILITY", "WORK_EXPERIENCE"]]
    leadership = [sent.text for sent in doc.sents if "leadership" in sent.text.lower()]
    technical = [sent.text for sent in doc.sents if "programming" in sent.text.lower() or "skills" in sent.text.lower()]
    
    return {
        "skills": skills[:5],  # Limit to top 5
        "leadership_sentences": leadership[:2],  # Limit to top 2
        "technical_sentences": technical[:2]  # Limit to top 2
    }

# GPT summarization function
def summarize_with_gpt(extracted_info, job_titles=None):
    """
    Generate a human-readable summary using OpenAI GPT based on extracted and TF-IDF info.
    """
    # Prepare the prompt
    prompt = f"""
    Based on the following extracted resume details, provide a concise summary highlighting what this person is good at and areas where they could improve. 
    If relevant, suggest improvements to match the following job titles: {', '.join(job_titles[:3]) if job_titles else 'None'}.

    Skills: {', '.join(extracted_info['skills'])}
    Leadership: {' '.join(extracted_info['leadership_sentences'])}
    Technical Skills: {' '.join(extracted_info['technical_sentences'])}

    Summary:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes resumes for recruiters."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Error generating summary with GPT: {e}")
        return "Error generating summary."
    
# complete the above instructions
def clean_resume(text):
    review = re.sub(
        '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
        " ",
        text,
    )
    review = review.lower()
    review = review.split()
    lm = WordNetLemmatizer()
    review = [
        lm.lemmatize(word)
        for word in review
        if not word in set(stopwords.words("english")).union(custom_stopwords)
    ]
    review = " ".join(review)
    return review

# Streamlit app layout
def main():
    st.title("Resume to Job Postings Matcher with TF-IDF, OpenAI Summarizer & Skill Highlights")
    

    # Load job and resume data
    jobs, resumes = load_data()

    # Sidebar Instructions
    st.sidebar.image("you_image.png")  # Add an image or logo for the dashboard
    st.sidebar.header("Instructions")
    st.sidebar.write("""
    1. Paste your resume into the text area to the right.
    2. The app will analyze your resume, summarize it, and find the most relevant job postings.
    3. Both your resume and the job descriptions will be highlighted with skills and entities.
    """)


    # Input resume
    st.subheader("Paste Your Resume")
    resume_input = st.text_area("Paste your resume here", height=300)
    # resume_input = clean_resume(st.text_area("Paste your resume here", height=300))

    if st.button("Analyze Resume"):
        if resume_input.strip():
            # Generate OpenAI embeddings for the input resume
            st.write("Generating embeddings for your resume...")
            resume_embedding = generate_openai_embedding(resume_input)

            if resume_embedding is not None:
                # Match input resume with job postings
                st.write("Matching your resume to job postings...")
                job_embeddings = np.vstack(jobs['description_embedding'])  # Stack job embeddings into 2D array
                job_titles = jobs['detail_title'].tolist()
                job_descriptions = jobs['cleaned_description'].tolist()
                job_urls = jobs['url'].tolist()  # Assuming 'job_url' column exists in the CSV

                top_matches = find_top_matches(resume_embedding, job_embeddings, job_titles, job_descriptions, job_urls)

                # Extract and summarize resume
                st.write("Summarizing your resume...")
                extracted_info = extract_resume_info(resume_input)
                human_summary = summarize_with_gpt(extracted_info, [match[0] for match in top_matches])

                # Display results
                st.markdown("### Your Resume with Highlighted Skills")
                highlighted_resume_html = highlight_skills_with_displacy(resume_input)
                st.markdown(highlighted_resume_html, unsafe_allow_html=True)

                st.markdown("### Human-Readable Resume Summary")
                st.write(human_summary)

                st.markdown("### Top Matching Job Descriptions")
                for i, (title, description, url, similarity) in enumerate(top_matches):
                    st.markdown(f"**Job {i+1}: [{title}]({url})** (Similarity: {similarity:.4f})")
                    highlighted_job_html = highlight_skills_with_displacy(description)
                    st.markdown(highlighted_job_html, unsafe_allow_html=True)
                    st.write("---")
        else:
            st.error("Please paste your resume into the text area before analyzing.")

if __name__ == "__main__":
    main()
