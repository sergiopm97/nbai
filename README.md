![Logo](https://raw.githubusercontent.com/sergiopm97/nbai/main/nbai_logo.png)

# NBAI

AI built to predict NBA games as accurately as possible 🤖🏀

## Features

- Predict the winners of NBA games for a specific date
- Return of predictions via API endpoint

## App setup

Clone the project

```bash
  git clone https://github.com/sergiopm97/nbai
```

Go to the project directory

```bash
  cd nbai
```

Create virtual environment

```bash
  python -m venv env
```

Activate the virtual environment

```bash
  & env/Scripts/Activate.ps1
```

Install the requirements in the virtual environment

```bash
  pip install -r requirements.txt
```

## Usage

First of all you need to launch the local server with the API running on it using the following script:

```bash
uvicorn main:app --reload
```

This command is going to start the server with the API in the http://127.0.0.1:8000 url. The active endpoint for the moment is the following one:

- http://127.0.0.1:8000/predictions/{date} -> specify a date and returns the NBA games predictions for that date (date format: YYYY-MM-DD)

If you want to retrain the predictive model because you have made changes in the training process, run the following script:

```bash
python .\pipeline.py
```

## Tech Stack

**Python version** -> 3.10.2

**Packages** -> Explore requirements.txt

## Authors

- [@sergiopm97](https://github.com/sergiopm97)
