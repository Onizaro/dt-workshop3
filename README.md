# create your model
to create a model, run:

```
python your_train_file.py
```

to run the server:

```
python app.py
```

to test the prediction:

```
curl "http://127.0.0.1:5000/predict?model_id=1&sex=male&age=22&fare=7.25&class=Third&sibsp=1&parch=0&embarked=Southampton"
```
