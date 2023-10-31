import sys
import os

#Modeling
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models
from src.utils import print_evaluvated_result
from src.utils import save_object
from src.utils import model_metrics


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def intiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting independent and dependent variable")
            xtrain,ytrain,xtest,ytest=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "GradientBoosting Regressor":GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict=evaluate_models(xtrain,ytrain,xtest,ytest,models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                logging.info("Best model has R2 score less than 60% ")
                raise CustomException("No best Model Found")
            print(f"Best Model Found, Model Name {best_model_name}, R2 score {best_model_score}")
            print('\n====================================================================================\n')
            logging.info(f"Best Model Found, Model Name {best_model_name}, R2 score {best_model_score}")
            logging.info("Hyperparameter Tunnig Started")

            ## Hyperparameter tunning in CatBosstRegressor
            ## Intializing CatBoostRegressor
            cbr=CatBoostRegressor(verbose=True)

            # creating Hyperparameter Grid
            param_dict={
                'depth' : [4,5,6,7,8,9, 10],
                'learning_rate' : [0.01,0.02,0.03,0.04],
                'iterations'    : [300,400,500,600]
            }
            ## Intializing Rnadom Search Object
            rscv=RandomizedSearchCV(cbr,param_dict,scoring=r2_score,cv=5,n_jobs=-1)
            ## Fit the model
            rscv.fit(xtrain,ytrain)
            ## Print the tunned parameter and score 
            print(f"Best CatBoost Parameter {rscv.best_params_}")
            print(f"Best CatBoost Score {rscv.best_score_}")
            print('\n====================================================================================\n')

            best_cbr = rscv.best_estimator_
            logging.info("Hyperparameter Tunning on CatBosst Regressor Completed")

            logging.info("Hyperparameter Tunning started for KNN")

            # Intializing KNN
            knn=KNeighborsRegressor()
            ## Parameter
            k_range=list(range(2,31))
            param_grid=dict(n_neighbors=k_range)

            # Fitting the cvmodel
            grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2',n_jobs=-1)
            grid.fit(xtrain, ytrain)

            # Print the tuned parameters and score
            print(f'Best KNN Parameters : {grid.best_params_}')
            print(f'Best KNN Score : {grid.best_score_}')
            print('\n====================================================================================\n')

            best_knn = grid.best_estimator_

            logging.info('Hyperparameter tuning Complete for KNN')

            logging.info('Voting Regressor model training started')

            # Creating Final Voting Regressor
            er=VotingRegressor([('cbr',best_cbr),('xgb',XGBRegressor()),('knn',best_knn)],weights=[3,2,1])
            er.fit(xtrain,ytrain)
            print('Final Model Evaluation :\n')
            print_evaluvated_result(xtrain,ytrain,xtest,ytest,er)
            logging.info('Votting Regressor Tranning Completed')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=er
            )
            logging.info('Pickle File saved')
            # Evaluvating Ensemble Regressor(Votting Regressor)
            ytest_pred = er.predict(xtest)

            mae, rmse, r2 = model_metrics(ytest, ytest_pred)
            logging.info(f'Test MAE : {mae}')
            logging.info(f'Test RMSE : {rmse}')
            logging.info(f'Test R2 Score : {r2}')
            logging.info('Final Model Training Completed')

            return mae, rmse, r2 
            
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)