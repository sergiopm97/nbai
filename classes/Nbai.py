import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


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
        self.X_train, self.X_test, self.y_train, self.y_test = [
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
        ]
        self.X_train_scaled, self.X_test_scaled = pd.DataFrame, pd.DataFrame

    def get_data(self, url: str) -> pd.DataFrame:
        return pd.read_csv(url)

    def select_sample(self, nba_matches: pd.DataFrame, season: str) -> pd.DataFrame:
        return nba_matches[nba_matches[season] >= 1980]

    def select_columns(self, nba_matches: pd.DataFrame, columns: list) -> pd.DataFrame:
        return nba_matches[columns]

    def drop_unfinished_matches(
        self, nba_matches: pd.DataFrame, scores: list
    ) -> pd.DataFrame:
        return nba_matches[
            (nba_matches[scores[0]].notna()) & (nba_matches[scores[1]].notna())
        ]

    def generate_winner_column(self, home_score: int, away_score: int) -> int:
        if home_score > away_score:
            return 0
        return 1

    def generate_total_points(self, home_score: int, away_score: int) -> int:
        return home_score + away_score

    def perform_train_test_split(
        self, nba_matches: pd.DataFrame, targets: list
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X = nba_matches.drop(targets, axis=1)
        y = nba_matches[targets]
        return train_test_split(X, y, test_size=0.20, random_state=42)

    def scale_features(
        self, features_train: pd.DataFrame, features_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        scaler = StandardScaler().fit(features_train)
        features_train_scaled = pd.DataFrame(
            data=scaler.transform(features_train), columns=[features_train.columns]
        )
        features_test_scaled = pd.DataFrame(
            data=scaler.transform(features_test), columns=[features_test.columns]
        )
        pickle.dump(scaler, open("predictors/scaler.pkl", "wb"))
        return features_train_scaled, features_test_scaled
