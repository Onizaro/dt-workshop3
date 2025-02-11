# Workshop 3 Dec-Tech


## Group members
- Noufel
- Anton
- Sivappiryan
- Nizar


## The dataset
We chose the Titanic dataset from Seaborn. Our goal is to predict whether a passenger of the Titanic survived based on certain parameters.



## Question1 
### Create your model

To create a model you can run (you'll have to install the required libraries):

```
python you_train.py
```

### Try the model

Then you can start the Flask server and predict using this :


```
python app.py
```

Then on your navigator: "http://127.0.0.1:5000/predict?model_id=4&sex=male&age=22&fare=7.25&class=Third&sibsp=1&parch=0&embarked=Southampton" for example.


## Question 2 
We created the client.py file where we use ngrok but if you want to try all the models and see the consensus you can use de "/consensus" path like so: 

```
http://127.0.0.1:5000/consensus?sex=male&age=22&fare=7.25&class=Third&sibsp=1&parch=0&embarked=Southampton
```

## Question 3
The weighting system is includued in the question 2 in the "/consensus" path. The outpuut should look like this:

```
{
  "consensus": {
    "prediction": "Pas survecu",
    "probabilities": {
      "not_survived": "85.08%",
      "survived": "14.92%"
    }
  },
  "individual_predictions": [
    {
      "model_id": "1",
      "prediction": "Pas survecu",
      "probabilities": {
        "not_survived": "91.90%",
        "survived": "8.10%"
      },
      "weight": 1.0
    }, 
    ...
```


## Question 4
For the models' balance, we created a test.py file where 30 tests are performed, and the balance is adjusted based on the results, allowing us to determine which model is the most performant. The results are stored in model_balances.json file.







