import streamlit as st
import requests
import os
import re
import sys
from dotenv import load_dotenv
from xml.etree import ElementTree
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import googleapiclient.discovery
import feedparser
from newspaper import Article
import lxml.html.clean
from newsapi import NewsApiClient 
import time 
import pycountry 
from pywebio.input import *
from pywebio.output import *
from pywebio.session import *
from PIL import Image
import requests
from io import BytesIO


import json 
import requests 
  
import streamlit as st 
from streamlit_lottie import st_lottie 
  

# Streamlit Config
st.set_page_config(page_title="Researcher Help", page_icon="💡", layout="wide")

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details in atleast 50 words, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

if "saved_papers" not in st.session_state:
    st.session_state["saved_papers"] = []

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Allow dangerous deserialization while loading FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])
    return response["output_text"]

def get_search_query(text):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Allow dangerous deserialization while l oading FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    user_question="suggest best topic on the basis of the text"
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])
    return response["output_text"]

def get_search_chain():

    prompt_template = """
    Provide the best topics on the basis of the text, so that it is easy to find on YouTube\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def search_youtube(search_queries, max_results=5):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey='AIzaSyCda3KH2EX2SglesdC9boDm5o_cOLVZX2I')

    all_videos = [] 

    for query in search_queries:
        request = youtube.search().list(q=query, part="snippet", type="video", maxResults=max_results)
        response = request.execute()

        for item in response.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            thumbnail = item["snippet"]["thumbnails"]["high"]["url"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            all_videos.append({"title": title, "thumbnail": thumbnail, "url": video_url})

    return all_videos 

NEWS_API_KEY = "30b8de5cf1c241f5903e113b2ca7564a"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# 🔹 Function to Fetch Latest Research News
def get_research_news():
    try:
        response = newsapi.get_everything(
            q=" OR ".join([
                "scientific discoveries", 
                "latest research", 
                "breakthrough technology","scientific discoveries",
                 "latest research",
                 "breakthrough technology",
    "AI advancements",
    "medical research",
    "space exploration",
    "quantum computing",
    "climate change research",
    "biotechnology innovations",
    "genetic engineering",
    "nanotechnology",
    "robotics research",
    "cancer treatment research",
    "deep learning advancements",
    "renewable energy research"
            ]),
            language="en",
            sort_by="publishedAt"
        )
        articles = response.get("articles", [])
        
        if not articles:
            return [{"title": "No recent research news found.", "description": "", "url": "", "image": ""}]
        
        return [
            {
                "title": article["title"],
                "description": article["description"][:200] + "...",
                "url": article["url"],
                "image": article["urlToImage"] or "https://via.placeholder.com/400",
            }
            for article in articles if article["title"]
        ]
    except Exception as e:
        st.error(f"Error fetching research news: {str(e)}")
        return []
# Sidebar Navigation
with st.sidebar:
    app = option_menu(
        menu_title="Navigation",
        options=["Home", "Research Page", "Chatbot", "Video Explanation", "Trending", "Saved Papers"],
        icons=["house-fill", "book", "chat-fill", "youtube", "trophy", "person-circle"],
        menu_icon="chat-text-fill",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "black"},
            "icon": {"color": "white", "font-size": "23px"},
            "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "--hover-color": "blue"},
            "nav-link-selected": {"background-color": "#02ab21"},
        },
    )

# Function to Search Research Papers
def search_arxiv_papers(query, num_results):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={num_results}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
            published = entry.find("{http://www.w3.org/2005/Atom}published").text
            link = entry.find("{http://www.w3.org/2005/Atom}id").text
            papers.append({"title": title.strip(), "summary": summary.strip(), "published": published, "link": link})
        return papers
    except Exception as e:
        st.error(f"Failed to fetch papers: {e}")
        return None
    
# Home Page
if app == "Home":
    st.title("Home")
    st.write("Make your study easy with the Research Helper.")
    url = requests.get( 
    "https://lottie.host/7020dc52-d67b-4455-94a0-a433d3488377/iiEigj5CBq.json") 
# Creating a blank dictionary to store JSON file, 
# as their structure is similar to Python Dictionary 
    url_json = dict() 
    if url.status_code == 200: 
      url_json = url.json() 
    else: 
      print("Error in the URL") 
    st_lottie(url_json,height=450,width=450) 
  
  

# Research Page
elif app == "Research Page":
    st.title("Research Papers")
    query = st.text_input("Enter the keyword or topic:")
    num_results = st.slider("Number of Results", 1, 20, 5)

    if query:
        papers = search_arxiv_papers(query, num_results)
        if papers:
            for i,paper in enumerate(papers):
                st.subheader(paper["title"])
                st.write(f"**Published on**: {paper['published']}")
                st.write(f"**Summary**: {paper['summary']}")
                st.write(f"[Read Full Paper]({paper['link']})")
                if st.button(f"💾 Save Paper {i+1}", key=f"save_{i}"):
                   st.session_state["saved_papers"].append(paper)
                   st.success(f"Saved: {paper['title']}")
                st.markdown("---")
        else:
            st.warning("No papers found for the given query.")

# Chatbot
elif app == "Chatbot":
    st.header("Get the answers to your question💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
# Other Tabs
elif app == "Video Explanation":
    st.title("Video Explanation")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")

            search_query = get_search_query(raw_text)
            search_query = str(search_query) if search_query else ""  # Ensure it's a string
            search_query = re.sub(r'\d+', '', search_query).strip()

            search_queries = [search_query]  

            videos = search_youtube(search_queries)

            print(f"🔍 Search Query: {search_query}")
            print(f"🔍 YouTube Videos Found: {videos}")

            st.subheader("🔍 Found YouTube Videos:")
            if videos:
                for video in videos:
                    st.markdown(f"### [{video['title']}]({video['url']})")
                    st.image(video["thumbnail"], width=200)
                    st.write("---")
            else:
                st.warning("No videos found. Try another query.")
elif app == "Trending":
    st.title("Updates in the field of Research")

    news_articles = get_research_news()

    for news in news_articles:
        st.subheader(news["title"])

        # ✅ Download and resize the image properly
        try:
            response = requests.get(news["image"])
            img = Image.open(BytesIO(response.content))  # Convert URL to image
            resized_img = img.resize((300, 200))  # Resize image (width, height)
            st.image(resized_img, caption="News Image", use_container_width=False)
        except Exception as e:
            st.warning(f"Could not load image: {e}")

        st.write(news["description"])
        st.markdown(f"[🔗 Read More]({news['url']})", unsafe_allow_html=True)
        st.write("---")
elif app == "Saved Papers":
    st.title("Saved Papers")
    if len(st.session_state["saved_papers"]) !=0:
      for paper in st.session_state["saved_papers"]:
        st.write(paper["title"])
        st.write(f"🔗 [Read Full Paper]({paper['link']})")
        st.markdown("---")
    else:
        st.write("NO PAPER IS SAVED FOR NOW!!")