import configparser
from classes import Nbai


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("config/config.ini")

    nbai = Nbai()

    nbai.nba_matches_df = nbai.get_data(config["NBA_DATABASE"]["url"])
