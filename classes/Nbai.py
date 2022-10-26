import pandas as pd


class Nbai:

    """
    Nbai object which contains the entire
    pipeline process cycle from data download
    to the development of predictive models
    """

    def __init__(self) -> None:
        self.nba_matches_df = pd.DataFrame

    def get_data(self, url: str) -> pd.DataFrame:
        return pd.read_csv(url)
