import configparser


if __name__ == "__main__":

    config = configparser.ConfigParser()

    config["NBA_DATABASE"] = {
        "url": "https://projects.fivethirtyeight.com/nba-model/nba_elo.csv"
    }

    config["COLUMNS"] = {
        "features": ["elo1_pre", "elo2_pre", "elo_prob1", "elo_prob2", "quality"],
        "targets": ["score1", "score2"],
        "generated_targets": ["total_points", "winner"],
        "sample_filter": "season",
    }

    with open("config/config.ini", "w") as config_file:
        config.write(config_file)
