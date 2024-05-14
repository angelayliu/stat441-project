# stat441-project
## Task Description
The final project for this course was to strive for the highest prediction score in a Kaggle competition against our classmates. Given a dataset of survey results, where each observation was a survey response from one person in Europe, our task was to predict "How is religion important in one's life" on a 4-point scale ranging from most to least important, or if there was no answer. The prediction score was calculated using multiclass log-loss of our model's predictions on an unlabelled dataset.
## Approach
After conducting trial-and-error with many classification learning algorithms, the approach which yielded the best prediction score in an efficient time frame was to stack an untuned multinomial gradient boosting model (implemented using LightGBM) with an untuned XGBoost model. The meta-model used for stacking was another XGBoost model, but this time tuned to achieve the best training score.
## Results
Achieved a grade of 90% from a rank of 11/39.
## Further Extensions
Using CatBoost to implement gradient boosting might yield better prediction scores as this framework is more suitable for categorical features, which close to 98% of the features were in this dataset. There is also the option to add a successfully tuned or semi-tuned neural net to the stack, as well as experiment with the weights of each model.
