import configparser
from classes import Nbai
import ast


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("config/config.ini")

    nbai = Nbai()

    nbai.nba_matches_df = nbai.get_data(config["NBA_DATABASE"]["url"])

    nbai.selected_columns_df = nbai.select_columns(
        nbai.nba_matches_df,
        ast.literal_eval(config["COLUMNS"]["features"])
        + ast.literal_eval(config["COLUMNS"]["targets"]),
    )
