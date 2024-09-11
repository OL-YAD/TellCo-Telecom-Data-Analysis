# TellCo-Telecom-Analysis

Project Overview

This project involves a comprehensive analysis of TellCo, an existing mobile service provider in the Republic of Pefkakia. The analysis is being conducted for a potential investor to determine whether TellCo is a viable acquisition target. The project encompasses user overview analysis, user engagement analysis, experience analytics, and user satisfaction analysis.
nOTE: This project is part of 10 academy's KAIM2 training week 2 challenge

## Business Objective

To analyze opportunities for growth and make a recommendation on whether TellCo is worth buying or selling. This will be achieved by analyzing a telecommunication dataset containing information about customers and their activities on the network.


## Project Structure

```plaintext

TellCo-Telecom-Analysis/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml   # GitHub Actions
├── .gitignore              # files and folders to be ignored by git
├── requirements.txt        # contains dependencies for the project
├── README.md               # Documentation for the projects
├── src/
│   └── __init__.py
├── notebooks/
│   ├── __init__.py
|   |──data_preprocessing.ipynb         # Jupyter notebook for data cleaning and processing 
|   ├── User_Overview_Analysis.ipynb    # Jupyter notebook for user overview analysis 
|   ├── User_Engagement_Analysis.ipynb  # Jupyter notebook for user engagement analysis
|   ├── experience_analytics.ipynb      # Jupyter notebook for user experience analysis
|   |── Satisfaction_Analysis.ipynb     # Jupyter notebook for user satisfaction analysis
│   └── README.md                       # Description of notebooks directory 
├── tests/
│   └── __init__.py
└── scripts/
    ├── __init__.py
    ├── database_connection.py      # script for loading the dummy data from database
    ├── data_processing.py          # Script data processing, cleaning and user overview analysis s
    ├── user_engagement.py          # script for user engagement analysis   
    ├── experience_analytics.py     # script for user experience analysis 
    |──  satisfaction_analysis.py   # script for user satisfaction analysis 
    └── README.md                   # Description of scripts directory
    
```
## Setup

1. Clone the repository:
   ```
   git clone https://github.com/OL-YAD/TellCo-Telecom-Data-Analysis.git
   cd TellCo-Telecom-Data-Analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Streamlit App

To run the Streamlit app locally:

1. Ensure you're in the project directory and your virtual environment is activated (if you're using one).

2. Run the following command:
   ```
   streamlit run app.py
   ```

## Data Sources

- Tellco Data Channel (XDR) Records- Contains information about user sessions, traffic data, and engagement    metrics.


## Contact

For any questions or feedback, please open an issue on this GitHub repository.