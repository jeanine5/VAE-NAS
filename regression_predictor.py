"""

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


class RregressionPredictor:
    def __init__(self, csv_filename='Training_CSV.csv'):
        self.df = pd.read_csv(csv_filename)
        self.X = self.df[['latent_dim', 'mid_layer']]

        self.y_loss = self.df['loss']
        self.y_ood = self.df['OOD']

        self.X_train, self.X_test, self.y_acc_train, self.y_acc_test, self.y_int_train, self.y_int_test = (
            train_test_split(self.X, self.y_loss, self.y_ood, test_size=0.2, random_state=42))

        self.regression_model = DecisionTreeRegressor()

        self.model_loss = DecisionTreeRegressor()
        self.model_ood = DecisionTreeRegressor()

    def train_models(self):
        """
        Train the linear regression model after preprocessing the data
        """
        self.model_loss.fit(self.X_train, self.y_acc_train)
        self.model_ood.fit(self.X_train, self.y_int_train)

    def evaluate_models(self):
        """
        Evaluate the performance of the regression model on the test set
        """
        y_loss_pred = self.model_loss.predict(self.X_test)
        y_ood_pred = self.model_ood.predict(self.X_test)

        mean_loss_pred = y_loss_pred.mean()
        mean_ood_pred = y_ood_pred.mean()

        return mean_loss_pred, mean_ood_pred

    def predict_performance(self, new_architecture):
        """
        Make predictions for the performance of a new architecture
        """

        # Prepare input data for prediction
        new_data = {'latent_dim': [new_architecture.latent_dim], 'hidden_sizes_mean': [new_architecture.mid_layer]}

        # Use trained regression models to predict performance metrics
        acc_pred = self.model_loss.predict(pd.DataFrame(new_data, columns=self.X.columns))
        int_pred = self.model_ood.predict(pd.DataFrame(new_data, columns=self.X.columns))

        return acc_pred.mean(), int_pred.mean()
