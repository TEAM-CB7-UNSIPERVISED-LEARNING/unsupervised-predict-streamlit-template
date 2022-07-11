"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from curses import raw
import select
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview","Team Profile","Data analysis", "Contuct Details","Imformation","Datasets","Glossary","feedback","map location"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("solution")
        st.write("we build this app to look for movies using collaborative filtering and contend basesd filtering recommendation sytems")
        st.image("https://www.researchgate.net/profile/Lionel-Ngoupeyou-Tondji/publication/323726564/figure/fig5/AS:631605009846299@1527597777415/Content-based-filtering-vs-Collaborative-filtering-Source.png")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    # Building Team Page
    if page_selection == "Team Profile":
        #Team name
        st.title("Team CB7")
        #team mates names
        st.write("Thembani maswengani - Team Learder -email t.maswanganyi101@gmail.com ")
        st.write("Katlego Maponya -Team Coordinator -email mr015khai@gmail.com")
        st.write("Keabetswe sebanyoni  - email Keamogetswe.Sibanyoni@fnb.co.za ")
        st.write("Charmain Nhlapho     - email Charmainlindelwa@gmail.com")
        st.write("Sydney Abrahams      - email sidneyabrahamsrsa@gmail.com")
        st.write("MR easy gym          - email bethuelmas94@gmail.com")
        st.image("http://images.clipartpanda.com/working-together-as-a-team-TeamEvent.jpg")
        
    #EDA
    if page_selection == "Data analysis":
        st.write("EXPLONATORY DATA ANALYSIS")
        st.write("This is the first step in EDA, we need to understand the variables in the data set, like what are the input variables and what are the output variables? Then we need to understand the type of the variables in the data set, like is it Integer? A float value? Or a String value? Lastly we need to analyze if the variables are continuous or categorical. Gender is an example of categorical variable while height is an example of continuous variable")
        #IMAGE
        st.image("https://st3.depositphotos.com/3116407/15347/i/950/depositphotos_153476170-stock-photo-data-analysis-concept-on-a.jpg")
        
    #imformation
    if page_selection == "Imformation":
        st.write("In today’s technology driven world, recommender systems are socially and economically critical to ensure that individuals can make optimised choices surrounding the content they engage with on a daily basis. One application where this is especially true is movie recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options. With this context, EDSA is challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed, based on their historical preferences.  ")
        #Image
        st.image("https://miro.medium.com/max/1400/0*GN1m1Gemqx4K_lv1")
        st.subheader("Raw data and Labels")
        if st.checkbox("check raw data"):
            st.write("movies.csv")
            st.write("ratings.csv")
            st.write("genome_scores.csv")
            st.write("genome_tags.csv")
            st.write("imdb_data.csv")
            st.write("tags.csv")
            st.write("test.csv")
            st.write("train.csv")
            
            st.write("take a look at the picture below, and in a nutshell understand the benefits of the reccomendatin system")
            st.image("https://raw.githubusercontent.com/TEAM-CB7-UNSIPERVISED-LEARNING/datasets/main/recommendation-system.webp?raw=true")
            
    #Contuct Deteils
    if page_selection =="Contuct Details":
        st.write("Contuct Details")
        st.write("contuct us for inquiries and one of our wonderfull consultands will attend to you as soon as possible")
        st.write("avilability of stuff to answer your calls", "during the week mon to friday is between 8:00 and 18:00", "and during the Weekends is between 8:00 and 13:00")
        
        st.image("https://www.seekpng.com/png/full/420-4208722_contactusbanner-contact-us-banner-images-png.png")
        st.write("Mail - TeamCB7@gmail.com")
        st.write("facebook - ClubCB7")
        st.write("Tweeter - @TEAMCB7")
        st.write("whatsap - 0814466788")
        st.write("mobile - 0792233448")
        if st.checkbox("check availability of staff"):
            st.write("monday - 08:00 to 18:00")
            st.write("Tuesday - 08:00 to 18:00 ")
            st.write("wednesday - 08:00 to 18:00")
            st.write("Thursday - 08:00 to 18:00")
            st.write("friday - 08:00 to 18:00")
            st.write("weekends - 08:00 to 13:00")
        
    #dataset page
    if page_selection == "Datasets":
        st.write("Datasets")
        st.image("https://intellipaat.com/mediaFiles/2018/07/Structured-Data-Vs.-Unstructured-Data.png")
        st.write("Raw Data Used that was used to test and train our models")
        if st.checkbox("check raw data"):
            st.write("movies.csv")
            st.write("ratings.csv")
            st.write("genome_scores.csv")
            st.write("genome_tags.csv")
            st.write("imdb_data.csv")
            st.write("tags.csv")
            st.write("test.csv")
            st.write("train.csv")
            
            
            st.write(" for our reccomendation systems to work, we needed this data, we trined it and tested it using our models and we reached succes")
            
    #Glossary page
    if page_selection == "Glossary":
        st.write("Glossary of words used that need further explanations")
        st.image("https://www.termcoord.eu/wp-content/uploads/2021/08/Multilingual-digital-terminology-today.png")
        if st.checkbox("checkbox"):
            st.write(" contend based filtereing - item data focused")
            st.write("collabarative filtereing - user data focused")
            st.write("Hybrid Recommendations - uses both item data and user data")
            
    #feedback page
    if page_selection == "feedback":
        st.write("Here at CB7 we appriciate you thoughts and feedback, let us know what you think about our product and service, thank you in advance")
        st.image("https://media.istockphoto.com/photos/your-feedback-matters-picture-id688306678")
        st.write ("feedback")
        st.text_area("Enter text below","enter your sentence/Enter your feedback")
        st.info('thank you for your feeback')
        option = st.radio("sselect a Response",("Posetive","Negative","Moderate","room for improment"))
        
        
        st.write("thank you for you time, make sure you use our service again for more accurate movie reccomendations")
    #map location
    if page_selection == "map location":
        st.write("map location")
        st.map("https://www.google.co.za/maps/@-25.7458176,28.1935872,12z")
        
        
    

    
if __name__ == '__main__':
    main()
