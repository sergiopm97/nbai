from fastapi import FastAPI
from classes import Nbai
import configparser
import ast


config = configparser.ConfigParser()
config.read("config/config.ini")

app = FastAPI()


@app.get("/predictions/{date}")
def get_predictions_by_date(date: str) -> dict:

    """
    Extract predictions of NBA game
    winners for a specific date
    """

    nbai = Nbai()

    nbai.nba_matches_df = nbai.get_data(config["NBA_DATABASE"]["url"])

    nbai.matches_by_date = nbai.get_matches_by_date(
        nbai.nba_matches_df, config["COLUMNS"]["date"], date
    )

    nbai.selected_columns_df = nbai.select_columns(
        nbai.matches_by_date, ast.literal_eval(config["COLUMNS"]["features"])
    )
