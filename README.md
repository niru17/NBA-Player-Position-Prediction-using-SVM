# NBA-Player-Position-Prediction-using-SVM
This Python script performs NBA player position prediction using Support Vector Machine (SVM) classification. It utilizes player statistics as features and predicts whether a player belongs to the "Guard" or "Forward" position based on their performance metrics.

Overview

The program reads NBA player statistics data from a CSV file (nba_stats.csv) and preprocesses it to extract relevant features and target labels. It splits the data into training and testing sets, trains an SVM classifier, and evaluates its performance using accuracy scores, confusion matrices, and cross-validation.

Requirements
- Python 3.x
- pandas
- scikit-learn (sklearn)
  
Data
1. nba_stats.csv: Contains NBA player statistics including Offensive Rebounds (ORB), Defensive Rebounds (DRB), Personal Fouls (PF), Turnovers (TOV), Assists (AST), Field Goals Attempted (FGA), Steals (STL), Three-Point Field Goals Attempted (3PA), Free Throws Attempted (FTA), Two-Point Field Goals (2P), Points (PTS), Three-Point Percentage (3P%), Field Goals (FG), and Free Throw Percentage (FT%).
2. dummy_test.csv: Contains dummy test data for evaluating model accuracy.

Usage

1. Data Preparation:

Ensure nba_stats.csv and dummy_test.csv files are available in the same directory as the script.

2. Environment Setup:

Install the required dependencies by running pip install pandas scikit-learn.

3. Run the Script:

- Execute the Python script to perform NBA player position prediction.
- Adjust the feature columns, SVM parameters, and test data as needed.

4. Interpret Results:

- Analyze training and validation set accuracies, confusion matrices, and cross-validation scores.
- Evaluate model performance on dummy test data to assess generalization ability.
  
Output

- Training and Validation Accuracy: Display the accuracy scores of the SVM classifier on the training and validation sets.
- Confusion Matrices: Visualize the confusion matrices for the training, validation, and dummy test data.
- Cross-Validation Scores: Calculate and display the cross-validation scores to assess model robustness.
- Dummy Test Accuracy: Evaluate the accuracy of the SVM classifier on the provided dummy test data.
