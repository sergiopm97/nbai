import configparser
import pandas as pd
from classes import Nbai
import ast


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("config/config.ini")

    pd.set_option("mode.chained_assignment", None)

    nbai = Nbai()

    nbai.nba_matches_df = nbai.get_data(config["NBA_DATABASE"]["url"])

    nbai.selected_sample = nbai.select_sample(
        nbai.nba_matches_df, config["COLUMNS"]["sample_filter"]
    )

    nbai.selected_columns_df = nbai.select_columns(
        nbai.selected_sample,
        ast.literal_eval(config["COLUMNS"]["features"])
        + ast.literal_eval(config["COLUMNS"]["targets"]),
    )

    nbai.finished_matches_df = nbai.drop_unfinished_matches(
        nbai.selected_columns_df, ast.literal_eval(config["COLUMNS"]["targets"])
    )

    nbai.finished_matches_df[
        ast.literal_eval(config["COLUMNS"]["generated_targets"])[0]
    ] = nbai.finished_matches_df.apply(
        lambda x: nbai.generate_total_points(
            x[ast.literal_eval(config["COLUMNS"]["targets"])[0]],
            x[ast.literal_eval(config["COLUMNS"]["targets"])[1]],
        ),
        axis=1,
    )

    nbai.finished_matches_df[
        ast.literal_eval(config["COLUMNS"]["generated_targets"])[1]
    ] = nbai.finished_matches_df.apply(
        lambda x: nbai.generate_winner_column(
            x[ast.literal_eval(config["COLUMNS"]["targets"])[0]],
            x[ast.literal_eval(config["COLUMNS"]["targets"])[1]],
        ),
        axis=1,
    )

    (
        nbai.X_train,
        nbai.X_test,
        nbai.y_train,
        nbai.y_test,
    ) = nbai.perform_train_test_split(
        nbai.finished_matches_df,
        ast.literal_eval(config["COLUMNS"]["targets"])
        + ast.literal_eval(config["COLUMNS"]["generated_targets"]),
    )

    nbai.X_train_scaled, nbai.X_test_scaled = nbai.scale_features(
        nbai.X_train, nbai.X_test
    )

    nbai.train_winner_model(
        nbai.X_train_scaled, nbai.y_train, nbai.X_test_scaled, nbai.y_test, "winner"
    )
