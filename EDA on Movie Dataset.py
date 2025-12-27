#!/usr/bin/env python
# coding: utf-8

# # 🎬 Movie Dataset – Exploratory Data Analysis (EDA)
# 
# ## Objective
# ###### The objective of this project is to explore and clean the movie dataset, identify patterns, detect outliers, and extract meaningful insights using statistical analysis and data visualization.
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Initial Data Inspection
# 
# In this step, we examine the structure of the dataset,
# check column names, data types, and sample records.
# 

# In[2]:


df = pd.read_csv("movies.csv")


# In[3]:


df.shape


# In[4]:


df.head(10)


# In[5]:


df.info()


# In[6]:


df.describe()


# Checking sum of null value per column

# In[7]:


df.isnull().sum()


# ###### Dropping null value rows 

# In[8]:


df.dropna(subset=['released'], inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df.dropna(subset=['released','score', 'votes', 'writer', 'star', 'country', 'runtime'], inplace=True)


# In[11]:


df.isnull().sum()


# ##### Replacing null value and Unrated value in rating column to Not Rated

# In[12]:


df['rating'].unique()


# In[13]:


df['rating'].fillna('Not Rated', inplace=True)
df['rating'].replace('Unrated', 'Not Rated', inplace = True)


# In[14]:


df['rating'].unique()


# In[15]:


df.isnull().sum()


# ##### Now, changing null value to Unknown

# In[16]:


df['company'].unique()


# In[17]:


df['company'].nunique()


# In[18]:


df['company'].fillna("Unknown", inplace=True)


# In[19]:


df.isnull().sum()


# ##### Now, checking correcting gross column

# In[20]:


df['gross'].unique()


# In[21]:


sns.boxplot(x=df['gross'])


# In[22]:


plt.hist(x=df['gross'], bins=5, color='red', edgecolor='yellow')


# ##### By this two we got to known that gross left skewed therefore we will use median to fill null value in gross column

# In[23]:


df['gross'].fillna(df['gross'].median(), inplace=True)


# In[24]:


df.isnull().sum()


# In[25]:


plt.hist(x=df['gross'], bins=5, color='red', edgecolor='yellow')


# ##### Now checking null values and filling them

# In[26]:


df['budget'].nunique()          


# In[27]:


sns.boxplot(x=df['budget'])


# In[28]:


plt.hist(x=df['budget'], bins=5, color='red', edgecolor='yellow')


# ##### By this to graph we can see these are left skewed therefore we use median to fill the null value

# In[29]:


df['budget'].fillna(df['budget'].median(), inplace=True)


# In[30]:


df.isnull().sum()


# ##### Release column contain release date as well as release country

# In[31]:


df[['released_date', 'release_country']] = df['released'].apply(lambda x: pd.Series(x.split('(')))


# In[32]:


df.head()


# In[33]:


df['release_country'] = df['release_country'].str.rstrip(')')
df.head()


# In[34]:


df['release_country'].unique()


# In[35]:


df['release_country'].isnull().sum()


# In[36]:


df.sample(20)


# In[37]:


df[df['release_country'] == 'Japan']


# In[38]:


df['released_date'] = df['released_date'].str.strip()
df['release_country'] = df['release_country'].str.strip()


# ##### Correcting release date column's data type

# In[39]:


df['released_date'] = pd.to_datetime(df['released_date'], errors='coerce')


# In[40]:


df.head()


# ##### Making 3 columns for release_day, release_month, release_year

# In[41]:


df['year_released'] = df['released_date'].dt.year


# In[42]:


df['month_released'] = df['released_date'].dt.month


# In[43]:


df['day_released'] = df['released_date'].dt.day


# In[44]:


df.head()


# In[45]:


df.drop('released', axis=1 ,inplace = True)


# In[46]:


df.head()


# In[47]:


df.info()


# ##### Checking datatypes and correcting according the requirement

# In[48]:


df.dtypes


# In[49]:


df['votes'] = df['votes'].astype('Int64')


# In[50]:


df.dtypes


# In[51]:


df['runtime'] = df['runtime'].astype('Int64')


# In[52]:


df.head()


# ##### Combining genre's values whose value count is less than 10 

# In[53]:


df["genre"].unique()


# In[54]:


df["genre"].value_counts()


# In[55]:


def combine_small_genres(df, column='genre', min_count=10):
 
    # Count genre frequency
    genre_counts = df[column].value_counts()
    
    # Identify genres to replace
    small_genres = genre_counts[genre_counts < min_count].index
    
    # Replace with 'Other'
    df[column] = df[column].replace(small_genres, 'Other')
    
    return df


# In[56]:


df = combine_small_genres(df, column='genre', min_count=10)


# In[57]:


df['genre'].value_counts()


# ##### Plotting Score

# In[58]:


df['score'].dtype


# In[59]:


sns.boxplot(x=df['score'])


# In[60]:


plt.hist(x=df['score'], bins=5, color='red', edgecolor='yellow')


# ### Average IMDB Score of Movies
# 
# ##### The distribution of IMDB scores is slightly skewed and contains outliers. Therefore, in addition to the mean, the median is also calculated as a robust measure of central tendency.

# In[61]:


# Mean (average IMDB score)
mean_score = df['score'].mean()

# Median IMDB score (robust to skewness and outliers)
median_score = df['score'].median()

print("Average (Mean) IMDB Score:", round(mean_score, 2))
print("Median IMDB Score:", round(median_score, 2))


# In[62]:


df['country'].unique()


# In[63]:


df['country'].value_counts()


# ##### Countries producing fewer than 50 movies were grouped into an 'Other' category to reduce noise and improve visualization clarity.

# In[64]:


country_counts = df['country'].value_counts()
small_countries = country_counts[country_counts < 50].index
df['country_grouped'] = df['country'].replace(small_countries, "Other")


# In[65]:


country_grouped_counts = df['country_grouped'].value_counts()


# In[66]:


country_grouped_counts.plot(kind='bar', 
                            figsize=(10,5), 
                            color=["darkblue", "mediumblue", "blue", "royalblue", 
                                   "cornflowerblue", "dodgerblue", "deepskyblue", 
                                   "skyblue", "lightskyblue", "lightblue"])
plt.title("Movie Production by Country (Countries < 50 grouped as Other)")
plt.xlabel("Country Name")
plt.ylabel("Number of Movie")
plt.xticks(rotation=45)


for i, v in enumerate(country_grouped_counts.values):
    plt.text(i, v+50, str(v), fontsize=12, color='black', ha='center')

plt.show()


# ###### The United States has produced the highest number of movies in the dataset.  After grouping countries with fewer than 50 movies into an "Other" category, the dominance of the United States remains clearly visible, indicating a strong concentration of movie production in a few major countries.
# 

# ### What is the distribution of movie runtime?

# In[67]:


df['runtime']


# In[68]:


plt.hist(df['runtime'], 
         color='green', 
         edgecolor = 'red', 
         bins=30)
plt.title('Histogram  of Runtime')
plt.xlabel('Runtime')
plt.show()


# ##### The distribution of movie runtime is **right-skewed**, with most movies having a runtime between **90 and 120 minutes**.  
# ##### A small number of movies have significantly longer runtimes, creating a long right tail and indicating the presence of outliers.  
# ##### This suggests that the **median runtime** is a more representative measure of central tendency than the **mean**.
# 

# In[69]:


plt.boxplot(df['runtime'])
plt.title("Boxplot for Movie Runtime")
plt.xlabel("Runtime (minutes)")
plt.show()


# ##### The boxplot confirms that movie runtime is **right-skewd**, with several high-end outliers representing very long movies.
# ##### Most movies have runtimes concentrated around the **19-20 minute** range, while a small number of movies exceed **200 minutes.**
# ##### Due to the presence of skewness and outliers, the **median runtime** is a more reliable measure of central tendency than the mean.

# ### Relationship between movie budget and gross revenue

# In[70]:


plt.scatter(df['budget'], df['gross'], marker='*', c='aqua')
plt.title("Budget V/S Gross")
plt.xlabel("Budget")
plt.ylabel("Gross")
plt.show()


# In[71]:


plt.scatter(df['budget'], df['gross'], marker='*', c='aqua')
plt.title("Budget V/S Gross")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Budget")
plt.ylabel("Gross")
plt.show()


# ##### Both the linear-scale and log-scale scatter plots show a positive relationship between movie budget and gross revenue. However, due to the large number of movies and overlapping budget values, the plots remain crowded even after log transformation. Despite this overplotting, the log-scale visualization still provides a clearer view of the overall trend by reducing the influence of extreme values. This indicates that while higher budgets generally tend to generate higher gross revenue, budget alone does not guarantee financial success.
# 

# ### Which genre has the highest average IMDB score?

# In[72]:


df[['genre', 'score']].head()


# In[73]:


genre_avg = df.groupby('genre')['score'].mean()


# In[74]:


genre_avg.head()


# In[75]:


genre_avg_sorted = genre_avg.sort_values(ascending=False)


# In[76]:


genre_avg_sorted.head()


# In[77]:


genre_counts = df['genre'].value_counts()

valid_genres = genre_counts[genre_counts >= 10].index


# In[78]:


valid_genres


# In[79]:


df_filtered = df[df['genre'].isin(valid_genres)]


# In[80]:


genre_avg_score = (
            df_filtered.groupby('genre')['score'].mean().sort_values(ascending=False)
)


# In[81]:


colors = colors = [
    'black', 'dimgray', 'darkslategray', 'teal',
    'darkgreen', 'olive', 'darkgoldenrod',
    'maroon', 'darkred', 'indigo'
]
plt.figure(figsize=(10,6))
genre_avg_score.head(10).plot(kind="bar", color=colors)
plt.xlabel('Genre')
plt.ylabel('Average IMDB score')
plt.title('Top Genre by Average IMDB score')
plt.xticks(rotation=45)
plt.show()


# ##### After filtering out genres with fewer than 10 movies to avoid bias, the analysis shows that certain genres consistently achieve higher average IMDB scores than others. While the top-ranked genre has the highest average rating, the differences across genres are relatively moderate, indicating that genre alone does not determine movie quality. Other factors such as direction, storytelling, and audience engagement also play an important role in influencing IMDB scores.
# 

# ### Relationship between the number of votes and IMDB score?

# In[82]:


df['votes'].describe()


# ##### The votes distribution is highly right-skewed with extreme outliers because a small number of movies receive extremely high vote counts, pulling the mean far above the median. So a logarithmic scale is appropriate for visualization.
# 

# In[83]:


plt.scatter(df['votes'], df['score'], marker="*", color='teal')
plt.title('Votes V/S Score')
plt.xlabel('Votes')
plt.ylabel('Score')
plt.show()


# In[84]:


plt.scatter(df['votes'], df['score'], marker="*", color='teal')

plt.title('Votes V/S Score')
plt.xlabel('Votes')
plt.ylabel('Score')

plt.xscale('log')
plt.yscale('log')
plt.show()


# ##### The analysis shows a positive relationship between the number of votes and IMDB score. Movies with fewer votes exhibit a wide range of ratings, indicating higher variability and uncertainty. As the number of votes increases, IMDB scores tend to become more stable and generally higher. The log-scale visualization provides a clearer view of this trend by reducing skewness in the vote distribution, highlighting that popular movies are more consistently well-rated, though votes alone do not fully determine movie quality.
# 

# ### How has the number of movie releases changed over time?

# In[85]:


movies_per_year = df['year_released'].value_counts()


# In[86]:


movies_per_year_sorted = movies_per_year.sort_index()


# In[87]:


movies_per_year_sorted


# In[88]:


plt.figure(figsize=(10,8))
plt.plot(movies_per_year_sorted, linestyle='-', marker='o', color='red')
plt.title("Number of Movies Released Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.grid(True)
plt.show()


# ##### The number of movie releases shows a clear increasing trend from the early 1980s, followed by a relatively stable pattern from the 1990s onward, with around 200 movies released per year. Minor year-to-year fluctuations are observed, which are expected in production data. The sharp decline in the most recent year is likely due to incomplete data rather than an actual drop in movie releases.
# 

# ### Which directors have the highest average IMDB scores?

# In[89]:


director_counts = df['director'].value_counts()


# In[90]:


valid_director = director_count[director_count >= 1]
valid_director


# In[ ]:


valid_directors = director_counts[director_counts >= 25].index

df_director_filtered = df[df['director'].isin(valid_directors)]

director_avg_score = (
    df_director_filtered
    .groupby('director')['score']
    .mean()
    .sort_values(ascending=False)
)


# In[ ]:


plt.figure(figsize=(10,6))
director_avg_score.head(10).plot(kind='barh', color='teal')
plt.xlabel('Average IMDB Score')
plt.ylabel('Director')
plt.title('Top Directors by Average IMDB Score (Min 25 Movies)')
plt.gca().invert_yaxis()
plt.show()


# ##### Directors with a larger body of work show relatively consistent average IMDB scores, suggesting that experience and creative style contribute to sustained movie quality.
# 

# ### What key factors influence movie success (IMDB rating & box office performance)?

# ##### Movie success is measured in terms of IMDB rating (quality) and gross revenue (financial performance).
# 

# ##### Based on the analysis, movie success appears to be influenced by multiple factors. Movies with higher vote counts tend to have more stable and higher IMDB ratings, indicating that audience engagement is associated with perceived quality. Higher budgets generally correspond to higher box office revenue, suggesting that greater financial investment increases earning potential, although it does not guarantee success. Additionally, certain genres show slightly higher average IMDB  scores, indicating that genre plays a moderate role in influencing movie ratings.

# ## Conclusion

# ##### This final analysis integrates all findings to understand the combined impact of popularity, investment, and creative factors on movie success.
# 

# In[ ]:




