import configparser
from classes import Nbai
import ast


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("config/config.ini")

    nbai = Nbai()

    nbai.nba_matches_df = nbai.get_data(config["NBA_DATABASE"]["url"])

    nbai.selected_sample = nbai.select_sample(nbai.nba_matches_df, "season")

    nbai.selected_columns_df = nbai.select_columns(
        nbai.selected_sample,
        ast.literal_eval(config["COLUMNS"]["features"])
        + ast.literal_eval(config["COLUMNS"]["targets"]),
    )

    nbai.finished_matches_df = nbai.drop_unfinished_matches(
        nbai.selected_columns_df, ast.literal_eval(config["COLUMNS"]["targets"])
    )
