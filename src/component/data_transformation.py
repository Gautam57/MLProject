import  sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTranformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_tranformer_object(self):

        '''
        This Function is responsible for data tranformation
        '''
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]
            )
            logging.info("Numerical column scaling completed")

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical column encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,numerical_columns),
                    ("Categorical",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_tranformer_object()

            target_column_name="math_score"
            numerical_columns=['reading_score', 'writing_score']

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_features_train_df=train_df[target_column_name]

            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_features_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_features_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_features_test_df)

            train_arr=np.c_[
                input_features_train_arr,np.array(target_features_train_df)
            ]
            test_arr=np.c_[input_features_test_arr,np.array(target_features_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
