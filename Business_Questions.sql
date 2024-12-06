	USE spotify;


# List the top 10 popular tracks based on the popularity score.

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



# What is the popularity vs time of track relations?            

	SELECT t.track_id,
           t.track_name,
		   ROUND(AVG(t.duration_ms)/60000, 2) as mins,
           ROUND(AVG(p.popularity), 1) as avg_popularity
	FROM
		dim_tracks as t
	INNER JOIN fact_track_popularity as p
    ON t.track_id = p.track_id
	GROUP BY t.track_id,t.track_name
	ORDER BY avg_popularity DESC;


# List the top 10 popular artists based on popularity score of their tracks and has released at least more than 5 songs in total

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

# Categorize the albums based on the number of songs and then calculate the overall popularity of those categories

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
   
# What is the average popularity of the songs based on key signatures?  

		SELECT 	key_signature, 
				ROUND(AVG(p.popularity),1) as avg_popularity
		FROM
                dim_tracks as t
        INNER JOIN fact_track_popularity as p
        ON t.track_id = p.track_id
        GROUP BY key_signature
        ORDER BY avg_popularity DESC;

# List the top-10 EPs/Albums

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
            
            
            
