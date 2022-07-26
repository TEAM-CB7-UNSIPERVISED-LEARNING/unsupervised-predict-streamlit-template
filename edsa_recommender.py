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
    page_options = ["Welcome","About","Search for a movie","Recommender System","Dataset", "Solution Overview","Team Profile","Data analysis", "Contact Details","Information","Glossary","Reviews","History",]
    
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
        st.title("Search for Movies")
        st.image(('resources/imgs/R.gif'), use_column_width=True)
        st.markdown('Please Refer to the About Machine Learning Page to learn more about the techniques used to recommend movies. If you decide not to use the recommender systems you can use this page to filter movies based on the rating of the movie , the year in which the movie was released and the genre of the movies. After you change the filter you will be left with movies that are specific to that filter used. Then when you scroll down you will see the movie name and the link to a youtube trailer of that movie. When you click the link ,you will see a page on youtube for that specific movie and you can watch the trailer and see if you like it. This is an alternative method to you if you are not satisfied with the recommender engine . Enjoy! ', unsafe_allow_html=True)
        # Movies
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
                
            st.image("https://miro.medium.com/max/1400/0*ckAOzr7BW6fhFeGK.jpg")
                


        


        
    #welcome page
    if page_selection == "Welcome":
        st.write('# Welcome To Our Movie Recommender')
        st.image('resources/imgs/minions.jpg', use_column_width=True)
        st.write('## ** * Where Movie Lovers Belong...* **')
        st.info("You are welcomed to become a member of the family of movie Enthusiasts, so relax you mind and take it easy")
        #st.image('resources/imgs/Movie-Show-GIF-960.gif', use_column_width=True)
        st.image("https://images.pexels.com/photos/2507025/pexels-photo-2507025.jpeg?cs=srgb&dl=pexels-quark-studio-2507025.jpg&fm=jpg")
        
    
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("solution")
        st.info("We build this app to look for movies using collaborative filtering and contend basesd filtering recommendation system")
        st.image("https://www.researchgate.net/profile/Lionel-Ngoupeyou-Tondji/publication/323726564/figure/fig5/AS:631605009846299@1527597777415/Content-based-filtering-vs-Collaborative-filtering-Source.png")
        if st.checkbox("Overview"):
            st.write("### CONTENT BASED FILTERING")
            
            st.info('Uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.')
        

            st.write('### COLLABORATIVE BASED FILTERING')
            
            st.info('builds a model from your past behavior (i.e. movies watched or selected by the you) as well as similar decisions made by other users.')

            st.image("https://camo.githubusercontent.com/a5f31b84eaab0fd04616953a3e8d62b28952ea1485b73240cee5290e5461dd2f/68747470733a2f2f6769746875622e636f6d2f5445414d2d4342372d554e534950455256495345442d4c4541524e494e472f64617461736574732f626c6f622f6d61696e2f315f3151615879447543764d6d635a4b542d456e7554412d32303438783832322e6a7065673f7261773d74727565")
            
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
        st.image("https://s3-us-west-2.amazonaws.com/prd-rteditorial/wp-content/uploads/2018/03/13153742/RT_300EssentialMovies_700X250.jpg")

    # Building Team Page
    if page_selection == "Team Profile":
        #Team name
        st.title("Team CB7")
        st.write("Team CB7 is an elite group of data scientists all who are dedicated in making all types of recommendation systems that will benefit the en-user of this app thier own setisfation with guaranteed results")
        #team mates names
        st.image("http://images.clipartpanda.com/working-together-as-a-team-TeamEvent.jpg")

        if st.checkbox("Team mates"):
            st.info("Thembani Maswengani, Team Learder")
            st.image("resources/imgs/thembani.JPEG",use_column_width=True)
            st.write("Github Account:thembani47 ")
            st.write("email: t.maswanganyi@gmail.com")
            st.write("Kaggle Account:Thembani Maswnganyi ")
            st.write("YouTube channel:Programming with Tief Everything ")
            st.write("")
            
            st.info("Katlego Maponya, Team Coordinator")
            st.image("resources/imgs/katlego.JPEG.jpg",use_column_width=True)
            st.write("Github Account:Katlego Maponya 015khai ")
            st.write("email: mr015khai@gmail.com")
            st.write("Kaggle Account:Katlego Maponya ")
            st.write("")
            
            st.info("Keamogetswe Sibanyoni")
            st.image("resources/imgs/kea.JPEG.jpg",use_column_width=True)
            st.write("Github Account: Keankie ")
            st.write("email: Keamogetswe.Sibanyoni@fnb.co.za")
            st.write("Kaggle Account: Keamogetswe Sibanyoni ")
            st.write("")
            
            st.info("Charmain Nhlapho")
            st.image("resources/imgs/chrmain.JPEG.jpg",use_column_width=True)
            st.write("Github Account:Charmain Nhlapho ")
            st.write("email: charmainlindelwa@gmail.com")
            st.write("Kaggle Account: Lindelwa Charmain")
            st.write("")
            
            st.info("Sydney Abrahams")
            st.image("resources/imgs/sydney.JPEG.jpeg",use_column_width=True)
            st.write("Github Account:sidney48  ")
            st.write("email: mailto:sidneyabrahamsrsa@gmail.com ")
            st.write("Kaggle Account: Sidney Abrahams ")
            st.write("")
            
            st.info("Bethuel Masango")
            st.image("resources/imgs/msizwane.JPEG.jpg",use_column_width=True)
            st.write("Github Account: 123Masangobethuel")
            st.write("email: bethuelmas94@gmail.com")
            st.write("Kaggle Account:Bethuelsizwemasango ")
            st.write("")
            
        
    #EDA
    if page_selection == "Data analysis":
        st.write("EXPLORATORY DATA ANALYSIS")
        if st.checkbox("check EDA"):
            st.info("This is the first step in EDA, we need to understand the variables in the data set, like what are the input variables and what are the output variables? Then we need to understand the type of the variables in the data set, like is it Integer? A float value? Or a String value? Lastly we need to analyze if the variables are continuous or categorical. Gender is an example of categorical variable while height is an example of continuous variable")
        
        #IMAGE
        #bargraph
            st.info("Bargraph")
            st.write("seaborn.barplot A bar plot represents an estimate of central tendency for a numeric variable with the height of each rectangle and provides some indication of the uncertainty around that estimate using error bars. Bar plots include 0 in the quantitative axis range, and they are a good choice when 0 is a meaningful value for the quantitative variable, and you want to make comparisons against it.For datasets where 0 is not a meaningful value, a point plot will allow you to focus on differences between levels of one or more categorical variables.It is also important to keep in mind that a bar plot shows only the mean (or other estimator) value, but in many cases it may be more informative to show the distribution of values at each level of the categorical variables. In that case, other approaches such as a box or violin plot may be more appropriate.")
            st.image("resources/imgs/rating.png")
        #piechart
        
            st.info("pichart")
            st.write("The fractional area of each wedge is given by x/sum(x). If sum(x) < 1, then the values of x give the fractional area directly and the array will not be normalized. The resulting pie will have an empty wedge of size 1 - sum(x).The wedges are plotted counterclockwise, by default starting from the x-axis.")
            st.image("resources/imgs/pichart.png")
            
        #histogram
            st.info("bargraph")
            st.write("From bar plot we got the Top 15 Years, we can see that from 2004, more movies were released. More than 1000 movies were released from 2002 going up to 2015. 2015, 2016, 2014 and 2017 has higher numbers of released movies, with a count of more than 2000 released movies.")
            st.image("resources/imgs/production.png")
            
            #wordcloud
        
            st.info("wordcloud")
            st.write("from the word cloud we can see 2015, 2016, 2014 have the most frequency, but this doesn't give us much to understand the year with most released movies, even though we can see that the Gen Z (2000s) released more movies compared to Millennials (1990s) movies. We can conclude that more movies are released in the 2000s as the streaming technology is also growing.")
            st.image("resources/imgs/cloud.png")
            
        #bargraph
            st.info("bargraph")
            st.write("User ID 72315 has more ratings compared to other users, the ratings of more than 12000, the second, third, fourth and number five on the plot above has ratings from 3050 - 3680, which is not even half of the first user. the sixth to number ten has 2000 ratings. this shows that users do rate movies.")
            st.image("resources/imgs/user.png")
            
        #wordcloud
            st.info("Wordcloud")
            st.write("No genres listed,Documentary, Comedy and Drama are popular keywords.Words like 'Drama', 'Romance', ect. are all over the wordcloud. This is because most films, regardless of the genre seem to have some element of drama")
            st.image("resources/imgs/cloud 2.png")
            
        #wordcloud
            st.info("Wordcloud")
            st.write("We can observe that Man, Girl and Love are larger then the rest, which informs us that they are the most popular title words.Wind, Hill, Sex are relatively small which tells us that they are relatively less popular than other title words.")
            st.image("resources/imgs/cloud 3.png")
            
            st.write("Content-Based Filtering")
            st.info("The content-based approach uses additional information about users and/or items. This filtering method uses item features to recommend other items similar to what the user likes and also based on their previous actions or explicit feedback. If we consider the example for a movies recommender system, the additional information can be, the age, the sex, the job or any other personal information for users as well as the category, the main actors, the duration or other characteristics for the movies i.e the items.")
            
            st.info("The main idea of content-based methods is to try to build a model, based on the available “features”, that explain the observed user-item interactions. Still considering users and movies, we can also create the model in such a way that it could provide us with an insight into why so is happening. Such a model helps us in making new predictions for a user pretty easily, with just a look at the profile of this user and based on its information, to determine relevant movies to suggest.")
            
            st.info("We can make use of a Utility Matrix for Content-Based Methods. A Utility Matrix can help signify the user’s preference for certain items. With the data gathered from the user, we can find a relation between the items which are liked by the user as well as those which are disliked, for this purpose the utility matrix can be put to best use. We assign a particular value to each user-item pair, this value is known as the degree of preference and a matrix of the user is drawn with the respective items to identify their preference relationship.")
            
            st.write("Challenges faced with Content-based filtering")
            
            st.info("Content-based methods seem to suffer far less from the cold start problem than collaborative approaches because new users or items can be described by their characteristics i.e the content and so relevant suggestions can be done for these new entities. Only new users or items with previously unseen features will logically suffer from this drawback, but once the system is trained enough, this has little to no chance to happen. Basically, it hypothesizes that if a user was interested in an item in the past, they will once again be interested in the same thing in the future. Similar items are usually grouped based on their features. User profiles are constructed using historical interactions or by explicitly asking users about their interests. There are other systems, not considered purely content-based, which utilize user personal and social data.")
            
            st.image("https://camo.githubusercontent.com/a5f31b84eaab0fd04616953a3e8d62b28952ea1485b73240cee5290e5461dd2f/68747470733a2f2f6769746875622e636f6d2f5445414d2d4342372d554e534950455256495345442d4c4541524e494e472f64617461736574732f626c6f622f6d61696e2f315f3151615879447543764d6d635a4b542d456e7554412d32303438783832322e6a7065673f7261773d74727565")
        
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
        st.info("Contact us for inquiries and one of our wonderfull consultands will attend to you as soon as possible")
        st.write("Availability of stuff to answer your calls", "during the week monday to friday is between 8:00 and 18:00", "and during the Weekends is between 8:00 and 13:00")
        
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
        st.image("https://i.pinimg.com/originals/e6/d2/27/e6d22708bcabc0b47b837a5da9794df0.gif")
    #dataset page
    if page_selection == "Dataset":
        st.info("Datasets")
        st.image("https://intellipaat.com/mediaFiles/2018/07/Structured-Data-Vs.-Unstructured-Data.png")
        st.info("Raw Data Used that was used to test and train our models")
        if st.checkbox("check raw data"):
            st.write("movies.csv")
            st.write("ratings.csv")
            st.write("genome_scores.csv")
            st.write("genome_tags.csv")
            st.write("imdb_data.csv")
            st.write("tags.csv")
            st.write("test.csv")
            st.write("train.csv")
            
            
            st.info(" For our recommendation systems to work, we needed this data, we trained it and tested it using our models and we reached success.")
            st.image("https://images.pexels.com/photos/265685/pexels-photo-265685.jpeg?cs=srgb&dl=pexels-pixabay-265685.jpg&fm=jpg")        
    #Glossary page
    if page_selection == "Glossary":
        st.write("Glossary of words used that need further explanations")
        st.image("https://agsd.org.uk/wp-content/uploads/2019/11/glossary-shutterstock-508906084.jpg")
        if st.checkbox("checkbox"):
            st.write(" contend based filtereing - item data focused")
            st.info('Uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.')
        
            st.write("collabarative filtereing - user data focused")
            st.info('builds a model from your past behavior (i.e. movies watched or selected by the you) as well as similar decisions made by other users.')
        
            st.write("Hybrid Recommendations - uses both item data and user data")
            
            st.write("Enlightennt about recmmender sytems background")
            
            st.write("Who invented recommendation system?")
            
            st.info("The Xerox Palo Alto Research Center (PARC) invented Tapestry to address this issue, and in so doing, created one of the first recommender systems.")
            
            st.write("The basics of recommender systems were founded by researches into cognition science [1] and information retrieval [2], and its first manifestation was the Usenet communication system created by Duke University in the second half of the 1970s [3], where users were able to share textual content with each other.")
            
            st.write("When were automated recommender systems first developed and deployed?")
            
            st.info("Abstract Since their introduction in the early 1990's, automated recommender systems have revolutionized the marketing and delivery of commerce and content by providing personalized recommendations and predictions over a variety of large and complex product offerings")
            
            st.image("https://i.pinimg.com/564x/62/98/c9/6298c9062d27ae97f168d4ac61f04d4a.jpg")
        
        st.info("Thank you for your time, make sure you use our service again for more accurate movie recommendations")
    #map location
    if page_selection == "map location":
        st.info("map location")
        st.image("https://th.bing.com/th/id/R.700eeb8c0fc6003653d2c6b62636ea07?rik=UM1w2kDnZLabkQ&riu=http%3a%2f%2fwww.orangesmile.com%2fcommon%2fimg_city_maps%2fjohannesburg-map-3.jpg&ehk=7Qk6U3f7VgXfTBvxDu32nZUOf%2fBcSqdtWlFT3NtB5RU%3d&risl=&pid=ImgRaw&r=0")
        
    #reviews page
    if page_selection == 'Reviews':
        st.info("Here at CB7 we appriciate your thoughts and feedback, let us know what you think about our product and service, thank you in advance.")
        st.image("resources/imgs/feedback-digital.jpg")
        st.title("Feel free to get in touch with us")
        st.markdown('''<span style="color:blue"> **Help us improve this app by rating it. Tell us how to give you a better user experience.** </span>''', unsafe_allow_html=True)
        @st.cache(allow_output_mutation=True)
        def get_data():
            return []
        st.write("Enter your text below")
        name = st.text_input("User name")
        inputs = st.text_input("Let us improve your user experience!!!")
        rate = st.slider("Rate us", 0, 5)
        option = st.radio("select a Response",("Posetive","Negative","Moderate","room for improvement"))
        if st.button("Submit"):
            get_data().append({"User name": name, "Suggestion": inputs,"rating":rate})
        #st.markdown('''<span style="color:blue"> **What other users said:** </span>''', unsafe_allow_html=True)
        #st.write(pd.DataFrame(get_data()))
        st.markdown('''<span style="color:blue"> **For any questions contact us here:** </span>''', unsafe_allow_html=True)
        st.markdown('TeamCB7@gmail.com')
        #st.image(('resources/imgs/our contact2.PNG'), use_column_width=True)
        
        
        st.info("Thank you for your time, make sure you use our service again for more accurate movie recommendations.")
        st.image("resources/imgs/pop corn.jpg")
    if page_selection == "History":
        st.write("History of recommender systems")
        st.info("A brief and superficial overview")
        if st.checkbox("checkbox"):
            st.write("Introduction")
            st.write("In our daily lives we are getting into decision-making situations countless times, often even unnoticed. What should we put on in the morning which is suitable for our daily program? Which menu to choose in the dining room? Which task should we perform first? Which school should we enroll our child to? We answer thousands of crucial or everyday questions like this in our lifetime.")
            
            st.write("In the past we have often asked experts or friends to help us in these decisions, but for some time we also have other possibilities available. Not only the librarian or bookstore attendant can help us to select our next book anymore, but even a website selling books (as well), such as Amazon. The videos suggested by YouTube are all based on our previous browsing and offer audiovisual content that pleases us with a relatively high hit rate.")
            
            st.write("Compared to the conventional solution, which is based on the suggestions of our friends, the site mentioned above makes its proposal based on the world’s largest video library. Since our friends’ collective insight is only a fraction of that, it has an unbeatable advantage. This way we can discover songs of bands we would most likely never have met in any other way. Since using this method not only our acquaintances can offer us content, but everyone in the world does so – unintentionally – through the recommender system.")
            
            st.image("https://www.onespire.net/wp-content/uploads/2021/04/recommender-systems-brief-history.jpg")
            
            st.write("However, it is important to state here that – as we will see later-, the recommender systems also have their limitations, so probably (and hopefully) in certain areas these will never know us as well as our friends and relatives. Considering these, the author of this post would like to avoid the impression that he is encouraging anyone to replace ordinary human relationships with recommender systems. We should consider these more as an opportunity, which can help us to decide and save time in certain – less personal – questions, or find entertainment which we may have never discovered without these systems.")
            
            st.info("An alternative definition")
            
            st.write("Recommender system: an information-filtering system supporting the user in a given decision making situation by narrowing the set of possible options and prioritizing its elements in a specific context. Prioritization can be based on the user’s explicitly or implicitly expressed preferences and also on the previous behavior of users with similar preferences.")
            
            st.info("The history of recommender systems")
            
            st.write("Information has always been an inseparable part of nature. When winter arrives, when birds are migrating, where to find fish and how to catch them; which berries can we eat and which bring certain death? Our ancestors have been passing on their experience even in ancient times. They were driven by the instinct to survive. They saw themselves in their offspring, and thus wanted, mostly unconsciously, to achieve their immortality. After all they didn’t struggle with their offspring to have them die after gorging on a handful of berries.Information could only recently, five millennia ago turn into power, with the invention of writing, which became the cornerstone of the modern world. Writing down words preserves the idea for a long time. They display the vibrations of the human soul in written form, thus creating a new form of eternal life.To understand the essence of information we had to wait until the 20th century. Claude Shannon has realized that the information content of a message depends solely on how much it differs from the average. Therefore, it is not closely related to the content or the length of the message, what only counts is whether it is unexpected or not. A politician on the take has no information content nowadays, but if it turns out that one is managing the public wealth trusted to him or her with proper care that is almost a sensation. If yesterday’s news were announced on the TV today that would have zero information content, except maybe the knowledge that the news editor has gone mad. Shannon has also discovered that information can be transformed to numbers, or precisely to two state systems, thus creating the basic unit of the new, information-based world order. Considering its content one bit is 0 or 1, yes or no, exists or not, but its power is that with the chain of these we can code anything. By digitizing things mankind has achieved that information became not only lasting, but can be stored in virtually endless quantities, and at the same time we are able to distribute it extremely fast. In parallel with the spreading of computers used for civilian purposes satisfying user expectations on a continuously growing scale has gotten into the focus of development companies and researchers.Behind the rapidly increasing popularity of machines extremely serious efforts lie, which has been made to reduce “friction” between man and machine. Increasingly comfortable solutions have been created because attempts have been made to understand people’s needs and personalize the services provided by computers.The basics of recommender systems were founded by researches into cognition science [1] and information retrieval [2], and its first manifestation was the Usenet communication system created by Duke University in the second half of the 1970s [3], where users were able to share textual content with each other. These were categorized into newsgroups and subgroups for easier search, but it was not directly built on or targeted the preferences of users.The first known such solution was the computer librarian Grundy, which first interviewed users about their preferences and then recommended books to them considering this information. Based on the information collected the system allocated the user into a stereotype group using a rather primitive method, thus recommending the same books to all persons in the same group. For more information about the results of Grundy’s solution and its popularity among users, see Rich’s 1979 article [1]. This approach may seem a little outdated today, but at the time it was a paradigm shift in automated services, since it was personalised. It is important to note that this milestone has not been reached by all webshops, even now.However, Grundy’s solution quickly gained a lot of critics in the scientific world. Nisbett and Wilson state that “people are very weak in the study and description of their own cognitive processes” [4]. According to their studies, people often highlight their attributes which make them stand out from the rest of a particular group, making stereotyping efforts more difficult. Of course, it could happen that we simply want to create a different image of ourselves.As Heli Vainio, the manager of one of Northern Europe’s largest shopping malls phrases a little harshly in her earlier interview: “People respond to questionnaires in a way that makes them look better. I don’t care about lies. I’m interested in facts.” [5]. That’s why she’s equipped her shopping centre with Wi-Fi equipment that can track visitors within and in the immediate vicinity of the building with an accuracy of 2 meters. The goal is that instead of the visitors their actions should talk.Basically two very different directions of recommender systems have evolved over time: collaborative filtering and content-based filtering. The former attempts to map (profile) the taste of users and offers content to them that users with similar preferences liked. The content-based filtering is about knowing the dimensions of the entity to be recommended (for example, a musical content recommendation system can consider the following dimensions: style, artist, era, orchestration, etc.) and the user’s preferences for these dimensions or characteristics. Thus, every time a user likes another song, this new information is added to their profile.The first example of collaborative filtering and also the origin of the term was the Tapestry system developed by Xerox PARC, which allowed its users to take notes and comment on the documents they were reading (initially in binary form: liking or disliking it). Therefore, users could not only use the content of the documents to manually narrow their search, but on the basis of notes and reviews from other users, which once reaching an appropriate number of users, was able to rank the thematic documents rather well on the basis of their relevance and usefulness [6].GroupLens [3], which started in 1992, was already able to make automated recommendations for Usenet articles if the user had already evaluated some articles in the system. In the following years, many thematic recommender sites were developed, such as Ringo developed at MIT, later the Firefly music recommender pages, or the BellCore movie proposer system.The first solution, which not only tried to encompass narrower topics, but no less than the Internet itself was Yahoo!, under a different name then. The two Stanford students created a thematic website catalog with indexed pages, which quickly gained popularity and meant easier searches for millions on the Internet, and based on the Alexa rankings it is still ranked as the fifth most visited website.The roots of content-based filtering have to be looked for in the information retrieval field, from which many techniques have been transferred. The first documented solution came from Emanuel Goldberg in the 1920s (if not including the Jaquard loom presented in 1801, the predecessor of the Hollerith punchcard), which was a “statistical machine” that attempted to automatically find documents stored on celluloid tapes by searching for patterns [7].In the 1960s, thanks to a team of researchers at the University of Cornwall gathering around Salton, a model for automatic indexing of texts has been created over nearly a decade, which forms the basis of the text mining processes we know today [8]. The procedure is very simple, documents are classified along certain predetermined criteria (dimensions) that are collected into a vector as indices . The more similar two documents are to each other, the smaller the angle of the vectors describing them are.The next milestone was the CITE online catalogue system developed in 1979 by Tamás Doszkocs for the National Library of Medicine, which not only allowed users to search for books by category, but sorted them by relevance based on search terms.Content-based filtering has gained raison d’étre relatively late in the 1990s as a division of information retrieval. The main reason for the delay was that creating a well-functioning content-based filtering system, even in a particular subject, is a huge challenge, since the task is nothing less than to “understand” the examined topic and the factors influencing the relationship of the users to the discipline.One of the first and quite successful research on this subject was the Music Genome Project in 1999, which aims to “understand” and capture music through its properties. To this end, more than 450 such properties have been revealed and their relations have been described using an algorithm. The basis of the procedure is that when the user likes a song, positive values are assigned to its specific properties (such as style, era, artist, orchestration, beat, etc.). Songs with similar properties will then be further promoted on the preference list and brought to the user’s attention. The huge advantage to collaborative filtering is that very little information is sufficient at startup, while the former unfortunately requires a lot of users and feedback to identify people with similar tastes. However, the disadvantage is that usually it can hardly, or not at all make recommendations outside the user’s music lists, since it is not building on the similarities between users, only the “understanding” the properties of music as an entity. Pandora Internet Radio, which has 250 million users, is based on this project even today [9].The first solution to combine collaborative and content-based filtering solutions was Fab [10] developed by Stanford students, presented in 1994. They point out that their objective with the hybrid system is to eliminate the disadvantages of the two procedures which became known by that time. Their model consists of two basic processes: first they collect content for specific topics (such as websites or articles about financial topics), then for each individual user they select those items collected from specific subjects which highly likely will interest them specifically and finally these contents will reach the user.Combining the two approaches can be conceived in several ways, one procedure can be embedded in another, as Fab’s example shows, or it is possible to give a joint recommendation as a result of the two procedures, as Netflix does. Netflix’s algorithm, CineMatch was the most successful recommender system for online movie sales in the early 2000s. It was a serious catalyst for such research and the scientific field – which only started its independent existence in the 90s – started rapidly to develop. The 2006 Netflix Award’s challenge was to create a recommender algorithm based on the 100 million film reviews made available by them, which makes recommendations at least 10% better than the results of CineMatch. The 1 million dollar prize in 2009 was awarded for a solution that included 107 different algorithms and mixed their recommendations depending on the circumstances [11]. We can’t omit the biggest example of online referral systems today, amazon.com, which recommends products to the user based on a cooperative filtering technique, taking into account previously browsed and purchased products and what they are currently viewing.This technique is now used by many webshops in order to improve their sales figures. The purpose of the recommender systems operating in the online customer space is to customize the storefront according to the current taste and need of the customers. This creates an almost unbeatable competitive advantage for online shops over traditional brick-and-mortar store purchases, whose only forte is that the product can be touched by hand or tried on. Nowadays, it is more typical that after a fitting in the shop the products are ordered online, or using the product return option are bought without tryout")
            
            st.info("References")
            
            st.write("[1] E. Rich (1979): User modeling via stereotypes, Cognitive Science, Vol. 3, No. 4,pp. 329–354.[2] M. Sanderson – W. B. Croft (2012): The History of Information Retrieval Research, Proceedings of the IEEE, Vol. 100, pp. 1444–1451.[3] P. Resnick – N. Iacovou – M. Sushak – P. Bergstrom – J. Riedl (1994): GroupLens: An open architechure for collaborative filtering of netnews, In Proceedings of the ACM Conf. Computer Support Cooperative Work (CSC),pp. 175-186.[4] R. E. Nisbett – T. D. Wilson (1977): Telling more than we can know: Verbal reports on mental processes, Psychological Review, Vol. 84, No. 3. pp. 231-259.[5] http://www.walkbase.com/blog/nordics-largest-shopping-centre-wi-fi-analytics-will-drive-our-marketing-decisions[6] D. Goldberg – B. Oki – D. Nichols – D. B. Terry (1992): Using Collaborative Filtering to Weave an Information Tapestry, Communications of the ACM, December, Vol. 35, No. 12, pp. 61-70.[7] M. Sanderson – W. B. Croft (2012): The History of Information Retrieval Research, Proceedings of the IEEE, Vol. 100, pp. 1444–1451.[8] G. Salton – A. Wong – C. S. Yang (1975): A vector space model for automatic indexing, Communications of the ACM, Vol.18, No.11, pp. 613-620.[9] M.H. Ferrara – M. P. LaMeau (2012): Pandora Radio/Music Genome Project. Innovation Masters: History’s Best Examples of Business Transformation. Detroit. pp. 267-270. Gale Virtual Reference Library.[10] M. Balabanovic ́ – Y Shoham (1997): Fab: Content-based, Collaborative Recommendation, Communications of the ACM, Vol.40, No.3, pp.66-72.[11] Recommendation system – https://en.citizendium.org/wiki/Recommendation_system[12] http://www.nytimes.com/2012/02/19/magazine/shopping-habits.html?_r=0[13] J. B. Schafer – J. A. Konstan – J. Riedl (2001): E-Commerce recommendation applications, Data Mining and Knowledge Discovery, Vol. 5, No. 1, pp. 115–153.")
            
            

    

    
if __name__ == '__main__':
    main()
