import pandas as pd


class Nbai:

    """
    Nbai object which contains the entire
    pipeline process cycle from data download
    to the development of predictive models
    """

    def __init__(self) -> None:
        self.nba_matches_df = pd.DataFrame
        self.selected_sample = pd.DataFrame
        self.selected_columns_df = pd.DataFrame
        self.finished_matches_df = pd.DataFrame

    def get_data(self, url: str) -> pd.DataFrame:
        return pd.read_csv(url)

    def select_sample(self, nba_matches: pd.DataFrame, season: str) -> pd.DataFrame:
        return nba_matches[nba_matches[season] >= 1980]

    def select_columns(self, nba_matches: pd.DataFrame, columns: list) -> pd.DataFrame:
        return nba_matches[columns]

    def drop_unfinished_matches(
        self, nba_games: pd.DataFrame, scores: list
    ) -> pd.DataFrame:
        return nba_games[
            (nba_games[scores[0]].notna()) & (nba_games[scores[1]].notna())
        ]
