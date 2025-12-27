# 🎬 Movie Dataset – Exploratory Data Analysis (EDA)

## 📌 Overview

This project performs exploratory data analysis on a movie dataset to understand  
factors influencing **IMDB ratings** and **box office performance**.

## 🎯 Objective
- Analyse patterns and trends in movie data.
- Identify factors affecting movie success
- Prepare insights for future predictive modeling

## 🛠️ Tools
- Python
- Pandas
- NumPy
- Matplotlib

## 🧹 Data Cleaning
- Converted numerical columns (`votes`, `budget`, `gross`, `runtime`) to appropriate numeric types  
- Handled missing values using **median imputation** for skewed distributions  
- Split release information into separate date and country columns  
- Removed extra spaces and unwanted characters from text fields  
- Filtered low-frequency categories to avoid biased analysis  

## 🔍 Key Analysis
- Distribution of IMDB scores and runtime
- Movie release over time
- Votes vs IMDB score
- Budge vs gross revenue (log scale)
- Genre and director impact on ratings

## 📊 Key Insights
- Higher votes counts are associated with more stable and higher IMDB rating 
- Higher budget generally lead to higher revenue, but do not guarantee success
- Genre and director influence ratings moderately
- Financial variables are naturally skewed

## 🧠 Conclusion
Movie success depends on a combination of popularity, financial investment and creative factors rather than a single variable.

## 👤 Author
Prathamesh Shirsat
