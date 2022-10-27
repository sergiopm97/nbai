import pandas as pd


class Nbai:

    """
    Nbai object which contains the entire
    pipeline process cycle from data download
    to the development of predictive models
    """

    def __init__(self) -> None:
        self.nba_matches_df = pd.DataFrame
        self.selected_columns_df = pd.DataFrame

    def get_data(self, url: str) -> pd.DataFrame:
        return pd.read_csv(url)

    def select_columns(self, nba_matches: pd.DataFrame, columns: list) -> pd.DataFrame:
        return nba_matches[columns]
