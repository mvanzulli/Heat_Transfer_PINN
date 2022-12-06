import pandas as pd 
import matplotlib.pyplot as plt

csv_file = "./../src/data/raw/heat_transfer_finn.csv"


# Plot a data frame
def plot_data_frame(csv_file):
    df = pd.read_csv(csv_file)
    df.plot()
    plt.show()

