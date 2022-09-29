from joblib import Parallel, delayed
import joblib
from training import *
import numpy as np
import pandas as pd

lm_from_joblib = joblib.load('model1.pkl')
lm2_from_joblib = joblib.load('model2.pkl')
svr_cv_model_from_joblib = joblib.load('svr_cv_model.pkl')
svr_tuned_from_joblib = joblib.load('svr_tuned_model.pkl')

y_pred_linear_regression = lm_from_joblib.predict(x_test)

y_pred_bacward_elimination = lm2_from_joblib.predict(x_test2)

y_pred_svr = svr_cv_model_from_joblib.predict(x_test)

y_pred_svr_tuned = svr_tuned_from_joblib.predict(x_test)


prediction = pd.DataFrame(y_pred_linear_regression, columns=['y_pred_linear_regression']).to_csv('prediction_linear_model.csv')
prediction2 = pd.DataFrame(y_pred_bacward_elimination, columns=['y_pred_bacward_elimination']).to_csv('prediction_linear_backward_elimination.csv')
prediction3 = pd.DataFrame(y_pred_svr, columns=['y_pred_svr']).to_csv('prediction_svr_model.csv')
prediction3 = pd.DataFrame(y_pred_svr_tuned, columns=['y_pred_svr_tuned']).to_csv('prediction_svr_model_tuned.csv')