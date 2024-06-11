# PGA_LIV Text Mining Project

This repository contains a text mining project focused on analyzing and predicting the sentiment from golf players press conferences at different events, particularly PGA and LIV tournaments. The project involves data scraping, preprocessing, and analysis using various Python tools and libraries.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Overview
The project aims to scrape data from golf-related websites, preprocess the text data, and apply machine learning models to make predictions. Key tasks include:
- Scraping data from player profiles and press conferences
- Cleaning and preprocessing text data
- Exploratory Data Analysis (EDA)
- Building predictive models

## Folder Structure
pga_liv/
│
├── datasets/
│ ├── Players_Links.csv
│ ├── Press_Conferences.csv
│
├── Cleaning_EDA.ipynb
├── Golf_Scrapper.ipynb
├── Intro_Txt_Final_Proj.pdf
├── Predictions.ipynb
├── Preprocessor.ipynb
├── Txt_Mining.ipynb
├── cleaning_functions.py
├── prepr_functions.py
├── prepro_classes.py
├── scraping_model.py
├── confusion.png
├── jon_las.png
├── README.md


## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/mgallon235/pga_liv.git
    ```

2. Navigate to the project directory:
    ```bash
    cd pga_liv
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Data Scraping:**
   - Run `Golf_Scrapper.ipynb` to scrape data from the specified sources.

2. **Data Preprocessing:**
   - Use `Preprocessor.ipynb` to clean and preprocess the scraped data.

3. **Exploratory Data Analysis:**
   - Explore the data using `Cleaning_EDA.ipynb`.

4. **Model Building and Prediction:**
   - Build and evaluate prediction models using `Predictions.ipynb` and `Txt_Mining.ipynb`.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.


