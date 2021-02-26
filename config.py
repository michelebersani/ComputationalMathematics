from pathlib import Path

path_prj        = Path(__file__).parents[0]
path_data       = path_prj  / Path('./data')
shuffled_csv    = path_data / Path('./shuffled_cup.csv')