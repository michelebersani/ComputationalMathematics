from pathlib import Path

path_prj        = Path(__file__).parents[0]
path_data       = path_prj  / Path('./data')
path_exps       = path_prj / Path('./experiments')
shuffled_csv    = path_data / Path('./shuffled_cup.csv')
experiments_csv = path_data / Path('./experiments.csv')