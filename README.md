# Explainable-AI

# Run The Code:

In order to check how our embedding based lime generated both relational and non-relational explanations, the following command format needs to be executed:


            python  explanations.py data query
            
            Here data is either review,topic or tweet
            
            query: For the the review data query range 1 to 1000, for the the topic data query range 3500 to 6999, and for the the tweet data query range 0 to 666
            
            For example: python  explanations.py topic 6510
            
            The query 6510 is truly predicted as medicine.
            
            The non-relational explanations can be visualized as follows:
            
            
            https://user-images.githubusercontent.com/25291998/139723201-1f61c89b-837e-43cc-84e9-39cbfd6b929d.JPG
            
            
            
            The relational explanation in the following describes that the query 6510 is connected to most of the queries that are also in same class medicine, 
            and therefore the sum of weights of these connected quriries is high.
            
            
            
            https://user-images.githubusercontent.com/25291998/139723644-5f219fb5-decb-4b2d-a416-83405e1de547.jpeg

            
# Packages need to be installed

Python==3.6

Bert

LIME

SHAP
