import configparser


if __name__ == "__main__":

    config = configparser.ConfigParser()

    config["SOCCER_DATABASE"] = {
        "url": "https://projects.fivethirtyeight.com/nba-model/nba_elo.csv"
    }

    with open("config/config.ini", "w") as config_file:
        config.write(config_file)
