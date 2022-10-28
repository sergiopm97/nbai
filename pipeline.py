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

    nbai.finished_matches_df["total_points"] = nbai.finished_matches_df.apply(
        lambda x: nbai.generate_total_points(
            x[ast.literal_eval(config["COLUMNS"]["targets"])[0]],
            x[ast.literal_eval(config["COLUMNS"]["targets"])[1]],
        ),
        axis=1,
    )

    nbai.finished_matches_df["winner"] = nbai.finished_matches_df.apply(
        lambda x: nbai.generate_winner_column(
            x[ast.literal_eval(config["COLUMNS"]["targets"])[0]],
            x[ast.literal_eval(config["COLUMNS"]["targets"])[1]],
        ),
        axis=1,
    )
