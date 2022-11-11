import ast
import configparser
import pickle

from fastapi import FastAPI, HTTPException

from classes import Nbai

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
    ).dropna()

    matches_index = nbai.selected_columns_df.index.to_list()

    if nbai.matches_by_date.empty:
        return HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "No NBA matches found for the specified date"},
        )

    features_scaler = pickle.load(open("predictors/scaler.pkl", "rb"))

    nbai.scaled_features_to_predict = nbai.scale_features_to_predict(
        nbai.selected_columns_df, features_scaler
    )

    winner_model = pickle.load(open("predictors/winner/model.pkl", "rb"))

    nbai.matches_probabilities = nbai.predict_matches_winner(
        nbai.scaled_features_to_predict, winner_model
    )

    nbai.matches_context = (
        nbai.nba_matches_df.iloc[matches_index][
            ast.literal_eval(config["COLUMNS"]["context"])
        ]
        .reset_index()
        .drop("index", axis=1)
    )

    nbai.matches_predictions = nbai.generate_predictions(
        nbai.matches_context, nbai.matches_probabilities
    )

    return nbai.matches_predictions.to_dict(orient="records")
