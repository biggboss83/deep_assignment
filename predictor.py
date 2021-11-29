import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class Predictor:
    def __init__(self, name, model, args=None):
        """
        Constructor   
        :param name:  A name given to your predictor
        :param model: An instance of your ANN model class.
        :param parameters: An optional dictionary with parameters passed down to constructor.
        """
        self.name_ = name
        self.model_ = model
        self.model_.load_state_dict(args)
        self.model_.eval()
        return

    def get_name(self):
        """
        Return the name given to your predictor.   
        :return: name
        """
        return self.name_

    def get_model(self):
        """
         Return a reference to you model.
         :return: a model  
         """
        #   Make sure to call input = input.to(device) on any input tensors that you feed to the model
        return self.model_

    def predict(self, info_company, info_quarter, info_daily, current_stock_price):
        #
        """
        Predict, based on the most recent information, the development of the stock-prices for companies 0-2.
        :param info_company: A list of information about each company
                             (market_segment.txt  records)
        :param info_quarter: A list of tuples, with the latest quarterly information for each of the market sectors.
                             (market_analysis.txt records)
        :param info_daily: A list of tuples, with the latest daily information about each company (0-2).
                             (info.txt  records)
        :param current_stock_price: A list of floats, with the with the current stock prices for companies 0-2.

        :return: A Python 3-tuple with your predictions: go-up (True), not (False) [company0, company1, company2]
        """

        # Labels for the columns
        labels_info = ['company', 'year',	'day',	'quarter',	'expert1_prediction',
                       'expert2_prediction',	'sentiment_analysis',	'm1',	'm2',	'm3',	'm4']
        labels_market_analysis = ['segment',	'year',	'quarter',	'trend']
        labels_market_segments = ['company',	'segment']
        labels_stock_prices = ['company',	'year',
                               'day',	'quarter',	'stock_price']

        labels = ['company', 'year', 'day', 'quarter', 'expert1_prediction', 'expert2_prediction',
                  'sentiment_analysis', 'm1', 'm2', 'm3', 'm4', 'stock_price', 'segment', 'trend']

        x_0 = list(info_daily[0]) + [current_stock_price[0],
                                     info_quarter[0][0], info_quarter[0][-1]]
        x_1 = list(info_daily[1]) + [current_stock_price[1],
                                     info_quarter[1][0], info_quarter[1][-1]]
        x_2 = list(info_daily[2]) + [current_stock_price[2],
                                     info_quarter[1][0], info_quarter[1][-1]]

        df = pd.DataFrame([x_0, x_1, x_2], columns=labels)
        df = pd.get_dummies(df, columns=["company", "segment"])
        df = df.drop(["year", "quarter", "day"], axis=1)

        scaler = StandardScaler()
        scaler.fit(df.values)

        X = torch.tensor(scaler.transform(df.values), dtype=torch.float)

        y_0, y_1, y_2 = self.model_(X)  # Forward pass

        return y_0.item() >= 0.5, y_1.item() >= 0.5, y_2.item() >= 0.5
