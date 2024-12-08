import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import traceback as tb
from datetime import datetime as dt
from sklearn.preprocessing import normalize

def assign_closest_genre_with_time_signature(tracks_df, genre_features_df):
    
    # Prepare genre feature vectors and time signature mappings
    genre_features = genre_features_df[['track_genre', 'energy', 'speechiness', 'acousticness', 'tempo']].set_index('track_genre')
    genre_vectors = genre_features.to_numpy()
    genre_time_signatures = genre_features_df.groupby('track_genre')['time_signature'].apply(list).to_dict()
    genre_names = genre_features.index.tolist()

    # Function to find the closest genre
    def find_closest_genre(row):
        # Extract track features and time signature
        track_features = row[['energy', 'speechiness', 'acousticness', 'tempo']].values
        track_time_signature = row['time_signature']
        original_genres = row['genre_list']

        # Filter valid genres based on original genres and matching time signatures
        valid_genres = [g for g in original_genres if g in genre_names and track_time_signature in genre_time_signatures.get(g, [])]
        if not valid_genres:
            return None, original_genres

        # Get indices of valid genres
        valid_indices = [genre_names.index(g) for g in valid_genres]

        # Calculate cosine similarity between track features and valid genres
        similarities = cosine_similarity(track_features.reshape(1, -1), genre_vectors[valid_indices])[0]
        closest_index = valid_indices[np.argmax(similarities)]

        # Assign the closest genre and update the remaining genres
        assigned_genre = genre_names[closest_index]
        

        return assigned_genre

    # Apply the function efficiently
    tracks_df['primary_genre'] = tracks_df.apply(lambda row: find_closest_genre(row) \
                                                 if len(row['genre_list']) > 1 else row['genre_list'][0], axis=1)

    return tracks_df


#MAIN FUNCTION
if __name__ == '__main__':

    try:

            start_time = dt.now()
            print("Spotify Data Load Start Time : {}".format(start_time))
        
        ################################################################################
        ## Data Loading and Cleaning
        ################################################################################

            print("Connecting to Kaggle to download the dataset")
            
            # Ensure your kaggle.json file is in ~/.kaggle or set the environment variables manually
            os.environ['KAGGLE_CONFIG_DIR'] = '/Users/avirup/.kaggle'
            # Authenticate Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            print("Authentication Successful. Downloading the relevant dataset.")
            
            # Define the dataset you want to download
            dataset = 'maharshipandya/-spotify-tracks-dataset'
            destination = '/Users/avirup/Desktop/Bootcamps/Iron Hack/Datacamp/Week-3/Labs/Project'
            
            # Download the dataset
            api.dataset_download_files(dataset, path=destination, unzip=True)
            
            # Load the downloaded CSV file into a Pandas DataFrame
            csv_file = os.path.join(destination, 'dataset.csv')
            
            data = pd.read_csv(csv_file,index_col=0)
            df = data.copy()

            print("Data Cleaning in Progress...")
            
            #Removing the row with no track names
            df_clean = df[df.track_name.isna() == False]
            
            #Remove duplicated rows
            df_clean = df_clean.drop_duplicates(keep='first').reset_index(drop=True)
            
            #Removing all the tracks with 0 time signature since it's not defined and replacing all 
            #the time signature value 1 with 4 since it is the common time signature
            df_clean = df_clean[df_clean['time_signature'] != 0]
            df_clean['time_signature'] = df_clean['time_signature'].replace(1,4)
            
            
            #Separating the features that define a genre
            genre_features = ['track_genre','time_signature','energy', 'speechiness', 'acousticness', 'tempo']
            
            #Select track genres and remove duplicates
            genre_df = df_clean[genre_features].drop_duplicates(keep='first').reset_index(drop=True)
            
            #Group to find certain avg features
            genre_grp_df = genre_df.groupby(['track_genre','time_signature']).agg({'energy':'mean',
                                                                                   'speechiness': 'mean',
                                                                                   'acousticness':'mean',
                                                                                   'tempo':'mean'}).reset_index()

            # Columns to standardize
            columns_to_normalize = ['energy', 'speechiness', 'acousticness', 'tempo']
            
            # Compute L2 norms for the selected columns
            l2_norms_gen = np.linalg.norm(genre_grp_df[columns_to_normalize], axis=1, keepdims=True)
            
            # Apply L2 normalization to the selected columns since it retains direction necessary for Cosine Similarity
            genre_grp_df[columns_to_normalize] = normalize(genre_grp_df[columns_to_normalize], norm='l2')
            
            #Retrieve Tracks Data by removing the genre column
            tracks_df = df_clean.iloc[:,:-1]
            
            #Remove duplicates from the track dataset
            tracks_df = tracks_df.drop_duplicates(keep='first').reset_index(drop=True)
            
            #Make a list of the track_id original genre_list
            original_genres_df = df_clean[['track_id','track_genre']].groupby('track_id').agg({'track_genre': list})\
                                                                     .reset_index().rename(columns={'track_genre':'genre_list'})
                
            #Rejoin it to the original data frame
            tracks_df = pd.merge(tracks_df,original_genres_df, on='track_id', how='left')
            
            
            #Mapping the key to the letter
            key_dict = {0:'C', 1:'C#/Db', 2:'D', 3:'D#/Eb', 4:'E', 5:'F', 6:'F#/Gb', 7:'G', 8:'G#/Ab', 9:'A', 10:'A#/Bb', 11:'B'}
            tracks_df['scale'] = tracks_df['key'].map(key_dict)
            
            tracks_df['key_signature'] = np.where(tracks_df['mode'] == 0, tracks_df['scale'] + " Minor", tracks_df['scale'] + " Major")

            # Compute L2 norms for the selected columns
            l2_norms_trk = np.linalg.norm(tracks_df[columns_to_normalize], axis=1, keepdims=True)
            
            # Apply L2 normalization to the selected columns since it retains the direction necessary for Cosine Similarity
            tracks_df[columns_to_normalize] = normalize(tracks_df[columns_to_normalize], norm='l2')
            
            tracks_df = assign_closest_genre_with_time_signature(tracks_df, genre_grp_df)

            # Denormalize the genre table and the track data frames
            tracks_df[columns_to_normalize] = tracks_df[columns_to_normalize] * l2_norms_trk
            genre_grp_df[columns_to_normalize] = genre_grp_df[columns_to_normalize] * l2_norms_gen

            ################################################################################
            ## Data Normalization
            ################################################################################
        
            print("Data Cleaning Completed. Data Normalization in Progress...")
        
            
            #Make a fact table that stores the popularity score of each track
            track_popularity_df = tracks_df[['track_id',
                                             'popularity']].groupby('track_id').mean().reset_index()
            
            track_popularity_df['popularity'] = track_popularity_df['popularity'].round(0).astype(int)
            
            #Creating a bridge table for the artists to tracks
            artists_df = tracks_df[['artists','track_id']]
            
            #Generate the rows for each artist from the string of artists
            artists_df.loc[:, 'artists'] = artists_df['artists'].str.split(';')
            artists_df = artists_df.explode('artists', ignore_index=True)
            
            # Generate unique IDs for each artist, starting from 1
            artists_df['artist_id'] = artists_df['artists'].astype('category').cat.codes + 1
            
            #Create a data frame for the bridge map of the artist and tracks
            artist_track_df = artists_df[['artist_id','track_id']].drop_duplicates().reset_index(drop=True)
            
            #Create a data frame for the artist ID and artist names in a new data frame
            artists_df = artists_df[['artist_id','artists']].drop_duplicates().reset_index(drop=True)\
                                                            .rename(columns={'artists':'artist_name'})
            
            #Select the relevant columns of the tracks dimension
            tracks_df = tracks_df[['track_id',
                                   'track_name',
                                   'album_name',
                                   'primary_genre',
                                   'duration_ms',
                                   'explicit', 
                                   'key_signature',
                                   'time_signature',
                                   'danceability', 
                                   'energy', 
                                   'key', 
                                   'loudness',
                                   'mode', 
                                   'speechiness', 
                                   'acousticness', 
                                   'instrumentalness', 
                                   'liveness',
                                   'valence', 
                                   'tempo']]
            
            tracks_df = tracks_df.drop_duplicates().reset_index(drop=True) 
            
            #Select only the columns to make the album dimension
            album_df = tracks_df[['album_name']].drop_duplicates().reset_index(drop=True)
            album_df['album_id'] = album_df['album_name'].astype('category').cat.codes + 1
            album_df = album_df[['album_id','album_name']]
            
            #Replace the album_name with the album_id in the tracks table
            tracks_df = pd.merge(tracks_df,album_df,on='album_name',how='inner')
            
            tracks_df = tracks_df[['track_id',
                                   'track_name',
                                   'album_id',
                                   'primary_genre',
                                   'duration_ms',
                                   'explicit', 
                                   'key_signature',
                                   'time_signature',
                                   'danceability', 
                                   'energy', 
                                   'key', 
                                   'loudness',
                                   'mode', 
                                   'speechiness', 
                                   'acousticness', 
                                   'instrumentalness', 
                                   'liveness',
                                   'valence', 
                                   'tempo']]

            print("Data Normalization Completed. Connecting to SQL Database and Loading Tables")

            ################################################################################
            ## Creation of Tables and Data Loading in the Spotify Database and Excel File
            ################################################################################
            
            # Create a connection string
            connection_string = "mysql+mysqlconnector://ironhack:123456@127.0.0.1/spotify"
            # Create the SQLAlchemy engine
            engine = create_engine(connection_string)
            
            artists_df.to_sql(name="dim_artists", con=engine, if_exists="replace", index=False)
            print("Table : dim_artists : {} Rows Loaded".format(artists_df.shape[0]))
            
            album_df.to_sql(name="dim_albums", con=engine, if_exists="replace", index=False)
            print("Table : dim_albums : {} Rows Loaded".format(album_df.shape[0]))
            
            tracks_df.to_sql(name="dim_tracks", con=engine, if_exists="replace", index=False)
            print("Table : dim_tracks : {} Rows Loaded".format(tracks_df.shape[0]))
            
            genre_grp_df.to_sql(name="dim_genre", con=engine, if_exists="replace", index=False)
            print("Table : dim_genre : {} Rows Loaded".format(genre_grp_df.shape[0]))
            
            artist_track_df.to_sql(name="artist_track_mapping", con=engine, if_exists="replace", index=False)
            print("Table : artist_track_mapping : {} Rows Loaded".format(artist_track_df.shape[0]))
            
            track_popularity_df.to_sql(name="fact_track_popularity", con=engine, if_exists="replace", index=False)
            print("Table : fact_track_popularity : {} Rows Loaded".format(track_popularity_df.shape[0]))
            
            print("DataFrames loaded successfully into the tables in spotify database.")

            print("Writing DataFrames into File for Tableau Dashboards")

            # Specify the output Excel file path
            output_file = "spotify_datasets.xlsx"
            
            # Write multiple data frames to different sheets in one Excel file
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                artists_df.to_excel(writer, sheet_name='dim_artists', index=False)
                album_df.to_excel(writer, sheet_name='dim_albums', index=False)
                tracks_df.to_excel(writer, sheet_name='dim_tracks', index=False)
                genre_grp_df.to_excel(writer, sheet_name='dim_genre', index=False)
                artist_track_df.to_excel(writer, sheet_name='artist_track_mapping', index=False)
                track_popularity_df.to_excel(writer, sheet_name='fact_track_popularity', index=False)
            
            print(f"Dataframes have been written to {output_file}")

            ################################################################################
            ## Generating BI Datasets to solve the questions
            ################################################################################
        
            print("BI Question Datasets Generation in progress...")

            # Question-1: List the top 10 popular tracks based on the popularity scores
            query = '''
                    WITH top_10 AS (
                          SELECT track_id, popularity, DENSE_RANK() OVER (ORDER BY popularity DESC) AS track_rank
                          FROM fact_track_popularity
                        )
                    SELECT 
                      trk.track_name, 
                      trk.primary_genre, 
                      GROUP_CONCAT(art.artist_name ORDER BY art.artist_name SEPARATOR ', ') AS artist_names,
                      t10.popularity, 
                      t10.track_rank
                    FROM top_10 AS t10
                    INNER JOIN dim_tracks AS trk
                      ON t10.track_id = trk.track_id
                    INNER JOIN artist_track_mapping AS map
                      ON trk.track_id = map.track_id
                    INNER JOIN dim_artists AS art
                      ON map.artist_id = art.artist_id
                    WHERE t10.track_rank <= 10
                    GROUP BY t10.track_id, trk.track_name, trk.primary_genre, t10.popularity, t10.track_rank
                    ORDER BY t10.track_rank;
                    '''
            
            pop_scores_df = pd.read_sql(sql=query, con=engine)

            # Question-2: What is the relationship between the tracks' length and popularity?
        
            query = '''
                       SELECT t.track_id, 
            		   ROUND(AVG(t.duration_ms)/60000, 2) as mins,
                       ROUND(AVG(p.popularity), 1) as avg_popularity
            	FROM
            		dim_tracks as t
            	INNER JOIN fact_track_popularity as p
                ON t.track_id = p.track_id
            	GROUP BY t.track_id
            	ORDER BY avg_popularity DESC;
                    '''
            
            time_vs_pop_df = pd.read_sql(sql=query, con=engine)

            # Question-3: List the top 10 popular artists based on popularity score and has released at least more than 5 songs in total        

            query = '''
                WITH artist_pop AS (
                SELECT art.artist_name, 
                       COUNT(DISTINCT pop.track_id) as num_songs,
                       ROUND(AVG(pop.popularity),0) as avg_popularity_score
                FROM fact_track_popularity AS pop
                INNER JOIN artist_track_mapping AS map
                  ON pop.track_id = map.track_id
                INNER JOIN dim_artists as art
                ON map.artist_id = art.artist_id
                GROUP BY art.artist_name),
                all_time_ranking as (
                SELECT artist_name, 
                       num_songs, 
                	   avg_popularity_score,
                       DENSE_RANK() OVER (ORDER BY avg_popularity_score DESC, num_songs DESC) as artist_rank
                FROM artist_pop
                WHERE num_songs > 5)
                SELECT artist_name, 
                       num_songs, 
                	   avg_popularity_score,
                       artist_rank
                FROM all_time_ranking
                WHERE artist_rank <= 10;
                    '''
            
            popular_artists_df = pd.read_sql(sql=query, con=engine)  

            # Question-4: Categorize the albums based on the number of songs and then calculate the overall popularity of those categories.       

            query = '''
                WITH album_popularity AS
                (SELECT alb.album_name, 
                       COUNT(trk.track_id) AS num_songs,
                       ROUND(AVG(pop.popularity),0) AS avg_popularity_score
                FROM dim_tracks as trk
                INNER JOIN fact_track_popularity as pop
                ON trk.track_id = pop.track_id
                INNER JOIN dim_albums as alb
                ON trk.album_id = alb.album_id
                GROUP BY alb.album_name),
                album_category_rank AS (
                SELECT album_name,
                       CASE WHEN num_songs = 1 THEN 'Single'
                            WHEN num_songs >= 2 AND num_songs <= 7 THEN 'EP'
                            WHEN num_songs >= 8 AND num_songs <= 15 THEN 'Standard Album'
                            WHEN num_songs >= 16 AND num_songs <= 20 THEN 'Deluxe Album'
                            WHEN num_songs >= 21 AND num_songs <= 30 THEN 'Double Album'
                            ELSE 'Compilations' END as album_category,
                	   num_songs,
                	   avg_popularity_score,
                	   DENSE_RANK() OVER (ORDER BY avg_popularity_score DESC) as album_rank
                FROM album_popularity)
                SELECT album_category,
                       COUNT(album_name) as num_albums,
                       ROUND(AVG(avg_popularity_score),0) as avg_pop_score
                FROM album_category_rank
                GROUP BY album_category
                ORDER BY avg_pop_score DESC;
                    '''
            
            popular_album_cat_df = pd.read_sql(sql=query, con=engine)

            # Question-5: What is the average popularity of the songs based on key signatures?       

            query = '''
                        SELECT key_signature, 
                        	   AVG(p.popularity) as avg_popularity
                        FROM
                        		dim_tracks as t
                        INNER JOIN fact_track_popularity as p
                        ON t.track_id = p.track_id
                        GROUP BY key_signature
                        ORDER BY avg_popularity DESC;
                    '''
            
            key_sign_df = pd.read_sql(sql=query, con=engine)

            # Question-6: List the top-10 EPs/Albums

            query = '''
                    WITH album_popularity AS
                        (SELECT alb.album_id,
                                alb.album_name, 
                                GROUP_CONCAT(DISTINCT art.artist_name ORDER BY art.artist_name SEPARATOR ', ') AS artists,
                                GROUP_CONCAT(DISTINCT trk.primary_genre ORDER BY trk.primary_genre SEPARATOR ', ') AS genres,
                                COUNT(DISTINCT trk.track_id) AS num_songs,
                                ROUND(AVG(pop.popularity),0) AS avg_popularity_score
                        FROM dim_tracks as trk
                        INNER JOIN fact_track_popularity as pop
                        ON trk.track_id = pop.track_id
                    	INNER JOIN artist_track_mapping AS map
                    	  ON trk.track_id = map.track_id
                    	INNER JOIN dim_artists AS art
                    	  ON map.artist_id = art.artist_id
                        INNER JOIN dim_albums as alb
                        ON trk.album_id = alb.album_id
                        GROUP BY alb.album_id,alb.album_name),
                        album_category_rank AS (
                        SELECT album_name,
                               artists,
                               genres,
                               CASE WHEN num_songs = 1 THEN 'Single'
                                    WHEN num_songs >= 2 AND num_songs <= 7 THEN 'EP'
                                    WHEN num_songs >= 8 AND num_songs <= 15 THEN 'Standard Album'
                                    WHEN num_songs >= 16 AND num_songs <= 20 THEN 'Deluxe Album'
                                    WHEN num_songs >= 21 AND num_songs <= 30 THEN 'Double Album'
                                    ELSE 'Compilations' END as album_category,
                        	   num_songs,
                        	   avg_popularity_score,
                        	   DENSE_RANK() OVER (ORDER BY avg_popularity_score DESC) as album_rank
                        FROM album_popularity)
                        SELECT album_name,
                               artists,
                               album_category,
                               genres,
                    		   album_rank as overall_rank,
                               CAST(SUM(num_songs) AS SIGNED) as num_songs,
                               ROUND(AVG(avg_popularity_score),0) as avg_pop_score
                        FROM album_category_rank
                        WHERE album_category <> 'Single'
                        GROUP BY album_name,artists,album_category,genres,album_rank
                        ORDER BY avg_pop_score DESC
                        LIMIT 10;
                    '''
            
            top_10_albums_df = pd.read_sql(sql=query, con=engine)

            print("BI Question Datasets Generation Completed. Writing into excel...")

            # Business Questions Datasets
            output_file = "business_questions_dataset.xlsx"
            
            # Write multiple data frames to different sheets in one Excel file
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                pop_scores_df.to_excel(writer, sheet_name='pop_scores', index=False)
                time_vs_pop_df.to_excel(writer, sheet_name='track_vs_pop', index=False)
                popular_artists_df.to_excel(writer, sheet_name='top_10_artists', index=False)
                popular_album_cat_df.to_excel(writer, sheet_name='album_cat_dist', index=False)
                key_sign_df.to_excel(writer, sheet_name='key_signature_dist', index=False)
                top_10_albums_df.to_excel(writer, sheet_name='top_10_albums', index=False)
            
            print(f"BI Questions Datasets have been written to {output_file}")

            end_time = dt.now()
            print("Spotify_Data_Load_Script Completed:{}".format(end_time))
            duration = end_time - start_time
            td_mins = int(round(duration.total_seconds() / 60))
            print('The difference is approx. %s minutes' % td_mins)
     
    except Exception as e:
        
        error = "Spotify_Data_Load_Script Failure :: Error Message {} with Traceback Details: {}".format(e,tb.format_exc())        
        print(error)


