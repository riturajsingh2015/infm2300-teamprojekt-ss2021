"""
This Connect4 class is a utility class for working with dataset from 
http://archive.ics.uci.edu/ml/datasets/Connect-4
It provides reading and transforming csv data into a pandas or numerical dataframe/array.
Furthermore it can visualize any row of the dataset as a grid like the game board.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Connect4:
    
    def __init__(self, csvPath):
        """
        Initializes Connect4 Utility class.
    
        Parameters
        ----------
        csvPath : path to connect-4.data. Could be relative or absolut
    
        Returns
        -------
        Instance of Connect4
        """
        self.dataframe = pd.read_csv(csvPath)

    def get_dataframe(self):
        """
        Returns the read data.
    
        Returns
        -------
        dataframe as pandas df
        """
        return self.dataframe

    def visualize_row(self, row_number):
        """
        Creates a matplot visualization like the plaing field of connect-4 game 
        of the row which was choosen.
        """
        # Take only columns with tile coordinates
        row = self.dataframe.iloc[row_number, :-1]
        # perform transformation on the row to resemeble connect4 board representation
        row=row.to_numpy()
        row=row.reshape((7,6)).T
        # flip horizontally
        row=np.flip(row, 0)
        # get indexs of 0 and x
        args_o=np.argwhere(row=='o')
        args_x=np.argwhere(row=='x')
        plt.title("Spielfeld")
        plt.axis([0, 7, 6, 0])
        for row_index,col_index in args_o:
            plt.text(col_index+(0.05),row_index+1-(0.05), r'o', fontsize=60)
        for row_index,col_index in args_x:
            plt.text(col_index+(0.05),row_index+1-(0.05), r'x', fontsize=60)
        plt.grid(True)

    def numerical_dataframe(self):
        """
        Returns a transformed dataframe from the original one.
        Since Strings are not suitable for neuronal networks.
        'b' -> 0
        'x' -> 1
        'o' -> 2
        'draw' -> 0
        'win' -> 1
        'loss' -> 2
    
        Returns
        -------
        dataframe as pandas df
        """
        numerical_dataframe = self.dataframe.to_numpy()
         
        # apply numeric transformations
        numerical_dataframe[numerical_dataframe=='x']=1
        numerical_dataframe[numerical_dataframe=='o']=2
        numerical_dataframe[numerical_dataframe=='b']=0
        numerical_dataframe[numerical_dataframe=='win']=1
        numerical_dataframe[numerical_dataframe=='loss']=2
        numerical_dataframe[numerical_dataframe=='draw']=0
        return numerical_dataframe.astype("uint8")
