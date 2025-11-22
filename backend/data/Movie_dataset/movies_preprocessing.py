import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from datetime import datetime


df = pd.read_csv("/Users/russell/Desktop/Fall2025/Ds1/Movie_dataset/movies_metadata.csv", low_memory=False)

# Convertion of numeric columns to numbers(integers)
numeric_columns = ['budget', 'popularity', 'revenue', 'runtime',
                   'vote_average', 'vote_count']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


#Null value replacement on numeric columns using the column mean
means = df[numeric_columns].mean()
df[numeric_columns] = df[numeric_columns].fillna(means)


#One-hot Encode the categorical column genre
def extract_genres(x):
    if pd.isna(x) or x == "":
        return []
    try:
        data = ast.literal_eval(x)
        return [d['name'] for d in data if 'name' in d]
    except:
        return []

df['genre_list'] = df['genres'].apply(extract_genres)

movie_genre = df['genre_list'].str.join('|').str.get_dummies()
df = pd.concat([df, movie_genre], axis=1)


# Binarization of original language column (Transform original_language to isEnglish, 1 if yes else 0)
df['isEnglish'] = (df['original_language'] == 'en').astype(int)

# Binarization of Boolean-like Columns

# Adult column (true/false)
df['adult'] = df['adult'].astype(str).str.lower().map({'true': 1, 'false': 0})
df['adult'] = df['adult'].fillna(0).astype(int)

# Belongs_to_collection column (either 1 or 0 if column is null)
df['collection?'] = df['belongs_to_collection'].notnull().astype(int)

# Video column
df['video'] = df['video'].astype(str).str.lower().map({'true': 1, 'false': 0})
df['video'] = df['video'].fillna(0).astype(int)


# Extract the release year from the release_date column and use it to determine the movie's age
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
current_year = datetime.now().year
df['movie_age'] = current_year - df['release_year']
df['movie_age'] = df['movie_age'].fillna(df['movie_age'].median())
df.loc[df['movie_age'] < 0 , 'movie_age'] = df['movie_age'].median()

# Standardize all the numeric columns
scale_cols = ['budget', 'popularity', 'revenue', 'runtime','movie_age']

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])


# Compute a weighted rating based on the vote_average and vote_count columns
C = df['vote_average'].mean()
m = df['vote_count'].quantile(0.80)  # top 20% vote counts

def weighted_rating(row, m=m, C=C):
    v = row['vote_count']
    R = row['vote_average']
    if v == 0:
        return C
    return (v / (v + m)) * R + (m / (v + m)) * C

df['weighted_vote'] = df.apply(weighted_rating, axis=1)



# Remove the columns from the dataset that do not provide useful information
columns_to_drop = [
    'homepage', 'imdb_id', 'poster_path', 'backdrop_path',
    'tagline', 'status','release_date','release_year',
    'spoken_languages', 'production_companies',
    'production_countries', 'genres', 'genre_list',
    'belongs_to_collection','original_language','id','original_title'
]

df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])


#Save preprocessed dataset
df.to_csv("movies_preprocessed.csv", index=False)
print("Preprocessing complete. Shape:", df.shape)
