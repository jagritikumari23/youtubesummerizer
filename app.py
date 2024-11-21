import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from gtts import gTTS  # For text-to-speech
import tempfile  # For temporary file management
from googletrans import Translator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import dropbox
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline  # For content categorization
import webbrowser

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Prompt template for summarization
prompt = """You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the content within the given time range or based on selected keywords.
Provide the important summary in points within 250 words. Please highlight key points.
Please provide the summary of the text given here: """

# Initialize translator and sentiment analyzer
translator = Translator()
analyzer = SentimentIntensityAnalyzer()

# Load pre-trained categorizer from Hugging Face
categorizer = pipeline("zero-shot-classification")

# Function to extract transcript details
def extract_transcript_details(youtube_video_url, desired_language="en"):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[desired_language])
        return transcript, desired_language
    except NoTranscriptFound:
        st.warning(f"No transcript available in {desired_language}. Trying auto-generated options...")
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            auto_transcript = transcript_list.find_generated_transcript(["hi", "en"])
            transcript = auto_transcript.fetch()
            return transcript, auto_transcript.language_code
        except Exception as fallback_error:
            st.error(f"Failed to fetch auto-generated transcript: {fallback_error}")
            return None, None
    except TranscriptsDisabled:
        st.warning("Transcripts are disabled for this video.")
        return None, None

# Translate transcript if needed
def translate_transcript(transcript, from_lang, to_lang="en"):
    full_text = " ".join([entry["text"] for entry in transcript])
    translation = translator.translate(full_text, src=from_lang, dest=to_lang)
    return translation.text

# Filter transcript by time range
def filter_transcript_time_range(transcript, start_time, end_time):
    return " ".join([entry["text"] for entry in transcript if start_time <= entry["start"] <= end_time])

# Filter transcript by keywords
def filter_transcript_keywords(transcript, keywords):
    filtered_text = ""
    for segment in transcript:
        if any(keyword.lower() in segment["text"].lower() for keyword in keywords):
            filtered_text += " " + segment["text"]
    return filtered_text

# Generate summary using Google Gemini Pro
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Format summary based on user selection
def format_summary(summary, format_type):
    points = summary.split("\n")
    formatted_summary = ""
    
    if format_type == "Bullet Points":
        for point in points:
            if point.strip():
                formatted_summary += f"- {point.strip()}\n"
    elif format_type == "Headings":
        for i, point in enumerate(points):
            if point.strip():
                formatted_summary += f"### {i+1}. {point.strip()}\n"
    
    return formatted_summary

# Convert text to podcast (audio)
def generate_podcast(summary_text):
    tts = gTTS(text=summary_text, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# Generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wordcloud

# Generate term frequency graph
def generate_term_frequency_graph(text):
    words = text.split()
    word_counts = Counter(words)
    df = pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False).head(10)
    return df

# Sentiment analysis function
def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment

# Content categorization function
def categorize_content(text):
    categories = ["Technology", "Education", "Entertainment", "Business", "Health", "Science"]
    result = categorizer(text, candidate_labels=categories)
    return result['labels'][0]  # Most likely category

# Streamlit App
st.set_page_config(page_title="YouTube Transcript to Podcast Converter", page_icon="🎧", layout="centered")

st.title("🎥 YouTube Transcript to Podcast Converter")
st.markdown("#### Transform your YouTube video transcripts into summarized podcasts!")

# Add user guide/tutorial section
st.markdown("## How to Use This App")
st.markdown("""
1. **Enter YouTube Video Link:** Paste the YouTube URL of the video you want to summarize.
2. **Select Time Range (Optional):** If you only want a summary from a specific part of the video, enter the start and end times in seconds.
3. **Filter by Keywords (Optional):** If you want to focus on certain topics, type keywords or phrases to filter the transcript.
4. **Choose Summary Formatting:** You can select how you want the summary to be presented: either as bullet points or in headings.
5. **Generate Notes & Podcast:** Click the button to generate a summary of the transcript, a podcast in MP3 format, a word cloud, and sentiment analysis.
6. **Download the Podcast:** After generating the podcast, you can download the MP3 file directly to your device.
7. **Provide Feedback:** Rate the quality of the summary and leave suggestions for improvement.
""")

# Video link input
youtube_link = st.text_input("Enter YouTube Video Link:")

# Time range input
st.markdown("### Select Time Range for Transcript (Optional)")
col1, col2 = st.columns(2)
start_time = col1.number_input("Start Time (in seconds):", min_value=0, step=1)
end_time = col2.number_input("End Time (in seconds):", min_value=0, step=1)

# Keywords input
st.markdown("### Filter Transcript by Keywords (Optional)")
keywords = st.text_input("Enter keywords or phrases, separated by commas:")

# Formatting style input
st.markdown("### Choose Summary Formatting Style")
format_type = st.selectbox("Select Formatting Style:", ["Bullet Points", "Headings"])

# Feedback storage
feedback_data = []

# Generate button
if st.button("Generate Notes & Podcast 🎙️"):
    if youtube_link:
        transcript, detected_language = extract_transcript_details(youtube_link)
        if transcript:
            if start_time or end_time:
                filtered_text = filter_transcript_time_range(transcript, start_time, end_time)
            elif keywords:
                keyword_list = [kw.strip() for kw in keywords.split(",")]
                filtered_text = filter_transcript_keywords(transcript, keyword_list)
            else:
                filtered_text = " ".join([entry["text"] for entry in transcript])
            
            if filtered_text:
                summary = generate_gemini_content(filtered_text, prompt)
                
                st.markdown("## Detailed Notes:")
                formatted_summary = format_summary(summary, format_type)
                st.markdown(formatted_summary)

                podcast_file = generate_podcast(summary)
                st.audio(podcast_file)
                st.download_button(
                    label="Download Podcast (MP3)",
                    data=open(podcast_file, "rb").read(),
                    file_name="podcast_summary.mp3",
                    mime="audio/mpeg",
                )
                os.remove(podcast_file)

                # Generate and display visualizations
                st.markdown("## Visualizations")
                
                # Word Cloud
                st.markdown("### Word Cloud")
                wordcloud = generate_wordcloud(filtered_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)

                # Term Frequency Graph
                st.markdown("### Term Frequency")
                term_freq_df = generate_term_frequency_graph(filtered_text)
                st.bar_chart(term_freq_df.set_index("Word"))

                # Sentiment Analysis
                sentiment = analyze_sentiment(filtered_text)
                st.markdown("### Sentiment Analysis")
                st.write(sentiment)

                # Content Categorization
                category = categorize_content(filtered_text)
                st.markdown(f"### Content Category: {category}")

                # Collect feedback
                st.markdown("### Feedback")
                rating = st.slider("Rate the quality of the summary (1 to 5):", 1, 5, 3)
                suggestion = st.text_area("Suggestions for improvement:")
                if st.button("Submit Feedback"):
                    feedback_data.append({"Rating": rating, "Suggestion": suggestion})
                    st.success("Thank you for your feedback!")
            else:
                st.warning("No content matches the specified filters.")
        else:
            st.warning("Failed to fetch transcript. Please check the video URL or transcript availability.")
    else:
        st.warning("Please enter a valid YouTube link to proceed.")

# Display collected feedback
if feedback_data:
    st.markdown("### Collected Feedback")
    feedback_df = pd.DataFrame(feedback_data)
    st.table(feedback_df)
















