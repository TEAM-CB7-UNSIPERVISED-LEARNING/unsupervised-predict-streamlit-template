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
    page_options = ["Search for a movie","welcome","About","Recommender System","Solution Overview","Team Profile","Data analysis", "Contact Details","Information","Datasets","Glossary","map location","Reviews","history","suscription"]
    
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
    if page_selection == "Search for a movie":
        st.info("search for a movie")
        st.image(('resources/imgs/countdown.gif'), use_column_width=True)
        st.markdown('Please Refer to the About Machine Learning Page to learn more about the techniques used to recommend movies. If you decide not to use the recommender systems you can use this page to filter movies based on the rating of the movie , the year in which the movie was released and the genre of the movies. After you change the filter you will be left with movies that are specific to that filter used. Then when you scroll down you will see the movie name and the link to a youtube trailer of that movie. When you click the link ,you will see a page on youtube for that specific movie and you can watch the trailer and see if you like it. This is an alternative method to you if you are not satisfied with the recommender engine . Enjoy! ', unsafe_allow_html=True)
        
        #movies
        df = pd.read_csv('resources/data/movies.csv')
        rating = pd.read_csv('resources/data/ratings.csv')
        
        def explode(df, lst_cols, fill_value='', preserve_index=False):
            import numpy as np
             # make sure `lst_cols` is list-alike
            if (lst_cols is not None
                    and len(lst_cols) > 0
                    and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
                lst_cols = [lst_cols]
            # all columns except `lst_cols`
            idx_cols = df.columns.difference(lst_cols)
            # calculate lengths of lists
            lens = df[lst_cols[0]].str.len()
            # preserve original index values    
            idx = np.repeat(df.index.values, lens)
            # create "exploded" DF
            res = (pd.DataFrame({
                        col:np.repeat(df[col].values, lens)
                        for col in idx_cols},
                        index=idx)
                    .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
            # append those rows that have empty lists
            if (lens == 0).any():
                # at least one list in cells is empty
                res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                            .fillna(fill_value))
            # revert the original index order
            res = res.sort_index()   
            # reset index if requested
            if not preserve_index:        
                res = res.reset_index(drop=True)
            return res 
        movie_data = pd.merge(rating, df, on='movieId')
        movie_data['year'] = movie_data.title.str.extract('(\(\d\d\d\d\))',expand=False)
        #Removing the parentheses
        movie_data['year'] = movie_data.year.str.extract('(\d\d\d\d)',expand=False)

        movie_data.genres = movie_data.genres.str.split('|')
        movie_rating = st.sidebar.number_input("Pick a rating ",0.5,5.0, step=0.5)

        movie_data = explode(movie_data, ['genres'])
        movie_title = movie_data['genres'].unique()
        title = st.selectbox('Genre', movie_title)
        movie_data['year'].dropna(inplace = True)
        movie_data = movie_data.drop(['movieId','timestamp','userId'], axis = 1)
        year_of_movie_release = movie_data['year'].sort_values(ascending=False).unique()
        release_year = st.selectbox('Year', year_of_movie_release)

        movie = movie_data[(movie_data.rating == movie_rating)&(movie_data.genres == title)&(movie_data.year == release_year)]
        df = movie.drop_duplicates(subset = ["title"])
        if len(df) !=0:
            st.write(df)
        if len(df) ==0:
            st.write('We have no movies for that rating!')        
        def youtube_link(title):
    
            """This function takes in the title of a movie and returns a Search query link to youtube
    
            INPUT: ('Avengers age of ultron')
            -----------
    
            OUTPUT: https://www.youtube.com/results?search_query=The+little+Mermaid&page=1
            ----------
            """
            title = title.replace(' ','+')
            base = "https://www.youtube.com/results?search_query="
            q = title
            page = "&page=1"
            URL = base + q + page
            return URL            
        if len(df) !=0:           
            for _, row in df.iterrows():
                st.write(row['title'])
                st.write(youtube_link(title = row['title']))


        


        
    #welcome page
    if page_selection == "welcome":
        st.title("'# Welcome To Our Movie Recommender")
        st.image("resources/imgs/minions.jpg" ,use_column_width=True)
        st.info('## ** * Where Movie Lovers Belong...* **')
    
    
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("solution")
        st.write("we build this app to look for movies using collaborative filtering and contend basesd filtering recommendation sytems")
        st.image("https://www.researchgate.net/profile/Lionel-Ngoupeyou-Tondji/publication/323726564/figure/fig5/AS:631605009846299@1527597777415/Content-based-filtering-vs-Collaborative-filtering-Source.png")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == 'About':
        st.title("About")
        st.info("Are you a movie lover? Are you tired of wasting your time watching tons of trailers and ending up not watching their movies? Are you tired of finishing your popcorns before you find the right movie? That has come to an end!!")
        st.image("resources/imgs/welcome.gif",use_column_width=True)
        st.markdown("Then we have got the right App for you.")
        st.subheader("How The App Works")
        st.info("The Movie Recommender App filters or predicts your preferences based on your favourite or watched movie selections. With just a few clicks, you will select three of your most favourite movies from thousands of movies on the app and you will get top 10 movies you are most likely to enjoy. You have an option to view some data visualizations including word clouds that show the most popular words that appear in movie titles and plots on the most popular genres. The app also contains a contact page, where users of the app can rate our app and give feedback and suggestions. Links to movie sites are also included, so the user has quick and easy to access the recommended movies.")
        st.subheader("Data Description")
        st.info("The dataset used for the movie recommender app consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB.")


    # Building Team Page
    if page_selection == "Team Profile":
        #Team name
        st.title("Team CB7")
        st.write("Team CB7 is an elite group of data scientists all who are dedicated in making all types of reccomendation systems that will benefit the en-user of this app thier own setisfation with guaranteed results")
        #team mates names
        if st.checkbox("Team mates"):
            st.info("Thembani Maswengani, Team Learder")
            st.write("Github Account: ")
            st.write("email: ")
            st.write("Kaggle Account: ")
            st.write("")
            
            st.info("Katlego Maponya, Team Coordinator")
            st.write("Github Account: ")
            st.write("email: ")
            st.write("Kaggle Account: ")
            st.write("")
            
            st.info("Keabetswe sebanyoni")
            st.write("Github Account: ")
            st.write("email: ")
            st.write("Kaggle Account: ")
            st.write("")
            
            st.info("Charmain Nhlapho")
            st.write("Github Account: ")
            st.write("email: ")
            st.write("Kaggle Account: ")
            st.write("")
            
            st.info("Sydney Abrahams")
            st.write("Github Account: ")
            st.write("email: ")
            st.write("Kaggle Account: ")
            st.write("")
            
            st.info("MR easy gym")
            st.write("Github Account: ")
            st.write("email: ")
            st.write("Kaggle Account: ")
            st.write("")
            
            st.image("http://images.clipartpanda.com/working-together-as-a-team-TeamEvent.jpg")
        st.image("https://media.istockphoto.com/photos/illustration-technology-hexagon-background-and-data-artificial-deep-picture-id1205195628")
        
    #EDA
    if page_selection == "Data analysis":
        st.write("EXPLONATORY DATA ANALYSIS")
        if st.checkbox("check EDA"):
            st.info("This is the first step in EDA, we need to understand the variables in the data set, like what are the input variables and what are the output variables? Then we need to understand the type of the variables in the data set, like is it Integer? A float value? Or a String value? Lastly we need to analyze if the variables are continuous or categorical. Gender is an example of categorical variable while height is an example of continuous variable")
        
        #IMAGE
        #bargraph
            st.info("Bargragh")
        
            st.image("https://www.kaggleusercontent.com/kf/100259092/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..qxugQq84DSt9HjTnmscbcw.DVAHyJFBuB9o9RsX6mKWKB3Fl1ABPgDVjSoPITW2k-pzIoxc4MFo3KsHUmks8boI0ufzI3e6wb1XX48lHDKnN1upMFSZhP39vMdJovjew5Bc8BGJfNdddLkON-Z19Dzhy1EPInnoDRfqvLDZAggbvJ7MzLp2-wd3LrulELjhFCSX6M6mMn6DEN_SlP4udMhSPwAQXNuEBiS5dZKRLwhqppLKj4xcQPsQlBHg292CnKqdoYIGvw5SUOdF7PCNy_iWRbaBzYVpULL5aWLZTMVtROMGkvhjZo7Lu3F_AlJdyNuJtJO_HtPUqtSiUHKtCay3RkPLxcMhPzasb-OusN1Vsj94Hw6OTw7-T_vG0LeEZZdXkPt7DhqVoxZJ4V9A3wxzyPsHKRzLNXR0cEB9DStMM-V55qt0e-o5769gvsBxME1L4XAobs50LjExi62dGUMkGABP1PQB56B1b-fgYftApcwQgkjQ_Oj93eEO6SwFSBDuTD3gvaMxbHJbKEUzo2XWe_vHLi660NcU5M1xsW3q1EHt5q4rbN_ryQWHE81RSjS0GAaBPd1AQDeyqkTkjzGcugxVssYMvk38AqpRHJ300FfyR1uwIV58FRGcCH6139MUHHdVQvocHdFoWNqSQqtfGq2cG0Ry8Z-Ldjovo6xhRBsChRE9cmpXfUvX42gMRWo.gL8HRJ_6S2WJSBTzWMD_uw/__results___files/__results___17_1.png")
        #piechart
        
            st.info("pichart")
        
            st.image("https://www.kaggleusercontent.com/kf/100259092/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..qxugQq84DSt9HjTnmscbcw.DVAHyJFBuB9o9RsX6mKWKB3Fl1ABPgDVjSoPITW2k-pzIoxc4MFo3KsHUmks8boI0ufzI3e6wb1XX48lHDKnN1upMFSZhP39vMdJovjew5Bc8BGJfNdddLkON-Z19Dzhy1EPInnoDRfqvLDZAggbvJ7MzLp2-wd3LrulELjhFCSX6M6mMn6DEN_SlP4udMhSPwAQXNuEBiS5dZKRLwhqppLKj4xcQPsQlBHg292CnKqdoYIGvw5SUOdF7PCNy_iWRbaBzYVpULL5aWLZTMVtROMGkvhjZo7Lu3F_AlJdyNuJtJO_HtPUqtSiUHKtCay3RkPLxcMhPzasb-OusN1Vsj94Hw6OTw7-T_vG0LeEZZdXkPt7DhqVoxZJ4V9A3wxzyPsHKRzLNXR0cEB9DStMM-V55qt0e-o5769gvsBxME1L4XAobs50LjExi62dGUMkGABP1PQB56B1b-fgYftApcwQgkjQ_Oj93eEO6SwFSBDuTD3gvaMxbHJbKEUzo2XWe_vHLi660NcU5M1xsW3q1EHt5q4rbN_ryQWHE81RSjS0GAaBPd1AQDeyqkTkjzGcugxVssYMvk38AqpRHJ300FfyR1uwIV58FRGcCH6139MUHHdVQvocHdFoWNqSQqtfGq2cG0Ry8Z-Ldjovo6xhRBsChRE9cmpXfUvX42gMRWo.gL8HRJ_6S2WJSBTzWMD_uw/__results___files/__results___18_0.png")
        
        #histogram
            st.info("histogram")
        
            st.image("https://www.kaggleusercontent.com/kf/100259092/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..qxugQq84DSt9HjTnmscbcw.DVAHyJFBuB9o9RsX6mKWKB3Fl1ABPgDVjSoPITW2k-pzIoxc4MFo3KsHUmks8boI0ufzI3e6wb1XX48lHDKnN1upMFSZhP39vMdJovjew5Bc8BGJfNdddLkON-Z19Dzhy1EPInnoDRfqvLDZAggbvJ7MzLp2-wd3LrulELjhFCSX6M6mMn6DEN_SlP4udMhSPwAQXNuEBiS5dZKRLwhqppLKj4xcQPsQlBHg292CnKqdoYIGvw5SUOdF7PCNy_iWRbaBzYVpULL5aWLZTMVtROMGkvhjZo7Lu3F_AlJdyNuJtJO_HtPUqtSiUHKtCay3RkPLxcMhPzasb-OusN1Vsj94Hw6OTw7-T_vG0LeEZZdXkPt7DhqVoxZJ4V9A3wxzyPsHKRzLNXR0cEB9DStMM-V55qt0e-o5769gvsBxME1L4XAobs50LjExi62dGUMkGABP1PQB56B1b-fgYftApcwQgkjQ_Oj93eEO6SwFSBDuTD3gvaMxbHJbKEUzo2XWe_vHLi660NcU5M1xsW3q1EHt5q4rbN_ryQWHE81RSjS0GAaBPd1AQDeyqkTkjzGcugxVssYMvk38AqpRHJ300FfyR1uwIV58FRGcCH6139MUHHdVQvocHdFoWNqSQqtfGq2cG0Ry8Z-Ldjovo6xhRBsChRE9cmpXfUvX42gMRWo.gL8HRJ_6S2WJSBTzWMD_uw/__results___files/__results___23_0.png")
        #wordcloud
        
            st.info("wordcloud")
        
            st.image("https://www.kaggleusercontent.com/kf/100259092/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..qxugQq84DSt9HjTnmscbcw.DVAHyJFBuB9o9RsX6mKWKB3Fl1ABPgDVjSoPITW2k-pzIoxc4MFo3KsHUmks8boI0ufzI3e6wb1XX48lHDKnN1upMFSZhP39vMdJovjew5Bc8BGJfNdddLkON-Z19Dzhy1EPInnoDRfqvLDZAggbvJ7MzLp2-wd3LrulELjhFCSX6M6mMn6DEN_SlP4udMhSPwAQXNuEBiS5dZKRLwhqppLKj4xcQPsQlBHg292CnKqdoYIGvw5SUOdF7PCNy_iWRbaBzYVpULL5aWLZTMVtROMGkvhjZo7Lu3F_AlJdyNuJtJO_HtPUqtSiUHKtCay3RkPLxcMhPzasb-OusN1Vsj94Hw6OTw7-T_vG0LeEZZdXkPt7DhqVoxZJ4V9A3wxzyPsHKRzLNXR0cEB9DStMM-V55qt0e-o5769gvsBxME1L4XAobs50LjExi62dGUMkGABP1PQB56B1b-fgYftApcwQgkjQ_Oj93eEO6SwFSBDuTD3gvaMxbHJbKEUzo2XWe_vHLi660NcU5M1xsW3q1EHt5q4rbN_ryQWHE81RSjS0GAaBPd1AQDeyqkTkjzGcugxVssYMvk38AqpRHJ300FfyR1uwIV58FRGcCH6139MUHHdVQvocHdFoWNqSQqtfGq2cG0Ry8Z-Ldjovo6xhRBsChRE9cmpXfUvX42gMRWo.gL8HRJ_6S2WJSBTzWMD_uw/__results___files/__results___24_0.png")
        
        #bargraph
            st.info("bargraph")
        
            st.image("https://www.kaggleusercontent.com/kf/100259092/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..qxugQq84DSt9HjTnmscbcw.DVAHyJFBuB9o9RsX6mKWKB3Fl1ABPgDVjSoPITW2k-pzIoxc4MFo3KsHUmks8boI0ufzI3e6wb1XX48lHDKnN1upMFSZhP39vMdJovjew5Bc8BGJfNdddLkON-Z19Dzhy1EPInnoDRfqvLDZAggbvJ7MzLp2-wd3LrulELjhFCSX6M6mMn6DEN_SlP4udMhSPwAQXNuEBiS5dZKRLwhqppLKj4xcQPsQlBHg292CnKqdoYIGvw5SUOdF7PCNy_iWRbaBzYVpULL5aWLZTMVtROMGkvhjZo7Lu3F_AlJdyNuJtJO_HtPUqtSiUHKtCay3RkPLxcMhPzasb-OusN1Vsj94Hw6OTw7-T_vG0LeEZZdXkPt7DhqVoxZJ4V9A3wxzyPsHKRzLNXR0cEB9DStMM-V55qt0e-o5769gvsBxME1L4XAobs50LjExi62dGUMkGABP1PQB56B1b-fgYftApcwQgkjQ_Oj93eEO6SwFSBDuTD3gvaMxbHJbKEUzo2XWe_vHLi660NcU5M1xsW3q1EHt5q4rbN_ryQWHE81RSjS0GAaBPd1AQDeyqkTkjzGcugxVssYMvk38AqpRHJ300FfyR1uwIV58FRGcCH6139MUHHdVQvocHdFoWNqSQqtfGq2cG0Ry8Z-Ldjovo6xhRBsChRE9cmpXfUvX42gMRWo.gL8HRJ_6S2WJSBTzWMD_uw/__results___files/__results___26_1.png")
        
        st.info("Source: The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB")
        st.image("https://img.freepik.com/premium-photo/stock-market-forex-trading-graph-graphic-concept_73426-102.jpg?w=1060")
        
    #imformation
    if page_selection == "Information":
        st.info("In today’s technology driven world, recommender systems are socially and economically critical to ensure that individuals can make optimised choices surrounding the content they engage with on a daily basis. One application where this is especially true is movie recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options. With this context, EDSA is challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed, based on their historical preferences.  ")
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
    if page_selection =="Contact Details":
        st.write("Contact Details")
        st.info("contact us for inquiries and one of our wonderfull consultands will attend to you as soon as possible")
        st.write("avilability of stuff to answer your calls", "during the week mon to friday is between 8:00 and 18:00", "and during the Weekends is between 8:00 and 13:00")
        
        st.image("https://www.seekpng.com/png/full/420-4208722_contactusbanner-contact-us-banner-images-png.png")
        st.info("Mail - TeamCB7@gmail.com")
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
            st.info("weekends - 08:00 to 13:00")
        
    #dataset page
    if page_selection == "Datasets":
        st.info("Datasets")
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
            
            
            st.info(" for our reccomendation systems to work, we needed this data, we trined it and tested it using our models and we reached succes")
            
    #Glossary page
    if page_selection == "Glossary":
        st.write("Glossary of words used that need further explanations")
        st.image("https://www.termcoord.eu/wp-content/uploads/2021/08/Multilingual-digital-terminology-today.png")
        if st.checkbox("checkbox"):
            st.write(" contend based filtereing - item data focused")
            st.info('Uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.')
        
            st.write("collabarative filtereing - user data focused")
            st.info('builds a model from your past behavior (i.e. movies watched or selected by the you) as well as similar decisions made by other users.')
        
            st.write("Hybrid Recommendations - uses both item data and user data")
            
    
        
        st.info("thank you for you time, make sure you use our service again for more accurate movie reccomendations")
    #map location
    if page_selection == "map location":
        st.info("map location")
        st.image("https://www.google.co.za/maps/@-25.7458176,28.1935872,12z")
        
    #reviews page
    if page_selection == 'Reviews':
        st.info("Here at CB7 we appriciate you thoughts and feedback, let us know what you think about our product and service, thank you in advance")
        st.image("https://media.istockphoto.com/photos/your-feedback-matters-picture-id688306678")
        st.title("feel free to get in touch with us")
        st.markdown('''<span style="color:blue"> **Help us improve this app by rating it. Tell us how to give you a better user experience.** </span>''', unsafe_allow_html=True)
        @st.cache(allow_output_mutation=True)
        def get_data():
            return []
        st.write("Enter your text below")
        name = st.text_input("User name")
        inputs = st.text_input("Let us improve your user experience!!!")
        rate = st.slider("Rate us", 0, 5)
        st.info("thank you in advance")
        option = st.radio("select a Response",("Posetive","Negative","Moderate","room for improment"))
        if st.button("Submit"):
            get_data().append({"User name": name, "Suggestion": inputs,"rating":rate})
        #st.markdown('''<span style="color:blue"> **What other users said:** </span>''', unsafe_allow_html=True)
        #st.write(pd.DataFrame(get_data()))
        st.markdown('''<span style="color:blue"> **For any questions contact us here:** </span>''', unsafe_allow_html=True)
        st.markdown('TeamCB7@gmail.com')
        #st.image(('resources/imgs/our contact2.PNG'), use_column_width=True)
        
        
        st.info("thank you for you time, make sure you use our service again for more accurate movie reccomendations")
        
        
        
    #suscription
    if page_selection == "suscription":
        st.info ("Price")
        option = st.radio("up 200","200 to 300","over 300")
        if st.checkbox("checkbox"):
            st.info("format")
            st.write("DVD")
            if st.checkbox("prices"):
                
                option = st.radio("select a price",("up to R150",'150 - 350', 'over 350'))
                st.write("Rmin -Rmax")
                
                st.info("Blue_ray")
                option = st.radio("select a price",("up to 150","150 t0 300","over 400"))
                st.write("Rmin - Rmax")
                
                st.info = ("series")
                option = st.radio("select a price",("R60", "R60 to R300"," over R300"))
                st.write("Rmin - Rmax")
                
                st.info= ("Drama")
                option = st.radio("select a price",("R40","R40 t0 R150","over R150"))
                st.write("Rmin -Rmax")
                
                st.info("MPAA Rating")
        
    

    
if __name__ == '__main__':
    main()
