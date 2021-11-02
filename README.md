# Explainable-AI

# Run The Code:

In order to get the overall accuracy results in Explanation_Accuracy folder  for explanation_analysis.py file the following coommands need to be run:

python explanation_analysis.py review # for the Yel Hotel review data

python explanation_analysis.py topic # for the 20NewsGroup data

python explanation_analysis.py tweet #  for the Covid-19 Tweet data






In order to check how our embedding based lime  generated both relational and non-relational explanations, the following command format needs to be executed
for explanations.py file in Explanation_visualization folder :


            python  explanations.py data query
            
            Here data is either review,topic or tweet
            
            query: For the the review data query range 1 to 1000, for the the topic data query range 3500 to 6999, and for the the tweet data query range 0 to 666
            
            For example: python  explanations.py review 150
            
            The query 150 is truly predicted as Positive review .
            
            The non_relational explanation will be saved as limetest_up.html file. 
            The non-relational explanations can be visualized as follows:
            
  
  ![image](https://user-images.githubusercontent.com/25291998/139792984-1ebe10d8-28c6-4ba2-930a-8f81e36faf43.png)


            
            
            
  The relational explanation in the following describes that the query 150 is connected to most of the queries that are also in same class Positive, 
  and therefore the percentage of weights of these connected queries is high.
  
  
  ![image](https://user-images.githubusercontent.com/25291998/139793091-96167b4b-6f0d-4750-a76d-3c40b6482aaf.png)


            

            
            
# Packages need to be installed

Python==3.6

Bert

LIME

SHAP
