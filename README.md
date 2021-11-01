# Explainable-AI

# Run The Code:

In order to check how our embedding based lime generated both relational and non-relational explanations, the following command format needs to be executed:


            python  explanations.py data query
            
            Here data is either review,topic or tweet
            
            query: For the the review data query range 1 to 1000, for the the topic data query range 3500 to 6999, and for the the tweet data query range 0 to 666
            
            For example: python  explanations.py topic 6510
            
            The query 6510 is truly predicted as medicine.
            
            The non-relational explanations can be visualized as follows:
            
            ![image](https://user-images.githubusercontent.com/25291998/139723874-427d22b8-d055-4a0f-af5c-062e9d2124cc.png)

            
            
            The relational explanation in the following describes that the query 6510 is connected to most of the queries that are also in same class medicine, 
            and therefore the sum of weights of these connected quriries is high.
            
            
           ![image](https://user-images.githubusercontent.com/25291998/139723917-2eda46a5-fc6b-4891-9757-cc929d1a3d24.png)
            
# Packages need to be installed

Python==3.6

Bert

LIME

SHAP
