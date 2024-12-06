# Project Objective 

The project aims to analyze Spotify tracks to answer some business questions.

# Dataset 

The dataset has been taken from Kaggle: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data

It has one CSV file where each row represents a track and its associated features and popularity

- Number of Rows in the dataset: 114,000
- Number of Columns in the dataset: 20

# Column Description

- **track_id**: The Spotify ID for the track
- **artists**: The artists' names who performed the track. If there is more than one artist, they are separated by a ;
- **album_name**: The album name in which the track appears
- **track_name**: Name of the track
- **popularity**: The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity.
- **duration_ms**: The track length in milliseconds
- **explicit**: Whether or not the track has explicit lyrics (true = yes it does; false = no it does not OR unknown)
- **danceability**: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm --stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable
- **energy**: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale
- **key**: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1
- **loudness**: The overall loudness of a track in decibels (dB)
- mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0
- **speechiness**: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks
- **acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic
- **instrumentalness**: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content
- **liveness**: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live
- **valence**: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)
- **tempo**: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration
- **time_signature**: An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4.
- **track_genre**: The genre in which the track belongs

# Data Cleaning

1. Check for nulls and remove them since a track must have a name and an artist.
2. Check for entire row-level duplicates and remove them.
3. Based on music theory, the time signature column had some wrong data with 0 and 1 time signatures. The ones with 0 were removed because there were only 160 rows(.001% of the total number of tracks), and the ones with 1 were changed to 4 because they represent the 4/4 time signature, which is also sometimes represented by 1.
4. While analyzing duplicates, it was found that there were multiple rows of the same track_id with different genres. To see the primary genre of the track, cosine similarity was performed on the features ('time_signature', 'energy', 'speechiness', 'acousticness', and 'tempo') between all the genres to that of the track.
5. Relevant Columns from this clean dataset were split further normalized into separate entities which is described in the next section. 

# ER-Relationships

After cleaning the data from the CSV file, some relationships between the attributes were stored in separate tables.

## Data Tables Summary

1. **Artists** - This stores the artist information - artists_df - 29,794 Artists
2. **Albums** - This stores the album information - album_df - 46,529 Albums
3. **Tracks** - This stores the track details - tracks_df - 89,578 Tracks
4. **Genre_Details** - This stores the genre details per time signature and its features - genre_df - 337 Genres
5. **Artist_Track_Map** - This stores the artist and the track mapping since one track can have multiple artists and one artist can have multiple tracks - artist_track_df
6. **Track_Popularity** - This stores the average popularity score of each track - track_popularity_df

## Python Script for End-to-End Process

The script **Spotify_Data_Loading.py** connects to Kaggle using API, downloads the relevant dataset into the local machine, and performs the following steps:

1. Data Cleaning by removing nulls, duplicates, and irrelevant columns.
2. Data Transformation of different columns based on the logic of music theory.
3. Data Normalization from one flat table and six different normalized tables.
4. Write the cleaned datasets into the tables created in the Spotify MySQL database.
5. Write the same datasets into an Excel file for Tableau Dashboard purposes due to the lack of MySql Connector in the Tableau Public version.
6. Execute the business question SQL queries and write them into another Excel file for further analysis and dashboard creation.

# Business Questions

- List the top 10 popular tracks based on the popularity score.

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


- What is the popularity vs time of track relations?
   
           SELECT  t.track_id,
        		   ROUND(AVG(t.duration_ms)/60000, 2) as mins,
                   ROUND(AVG(p.popularity), 1) as avg_popularity
        	FROM
        		dim_tracks as t
        	INNER JOIN fact_track_popularity as p
            ON t.track_id = p.track_id
        	GROUP BY t.track_id
        	ORDER BY avg_popularity DESC;


- List the top 10 popular artists based on the popularity score of their tracks and have released at least more than 5 songs in total

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

- Categorize the albums based on the number of songs and then calculate the overall popularity of those categories

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
   
- What is the average popularity of the songs based on key signatures?  

		SELECT 	key_signature, 
				ROUND(AVG(p.popularity),1) as avg_popularity
		FROM
                dim_tracks as t
        INNER JOIN fact_track_popularity as p
        ON t.track_id = p.track_id
        GROUP BY key_signature
        ORDER BY avg_popularity DESC;

- List the top-10 EPs/Albums

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
