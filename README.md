# NCAA Basketball Tournament Simulation

This project is a command-line tool for simulating the NCAA basketball tournament using a Markov chain model based on
team seeds and KenPom scores.

## Installation

Make sure you have Python 3.7 or higher installed. We recommend using a virtual environment for managing dependencies.

1. Clone the repository:

2. Create and activate a virtual environment (optional):

```bash
python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the simulation, pass a CSV file containing the team metrics (team name, seed, and KenPom score) as an argument to
the `main.py` script:

`python main.py teams.csv`

The CSV file should have columns named `team_name`, `team_seed`, and `ken_pom_score`. Here's an example:

```csv
team,seed,score
Team A,1,30.0
Team B,2,28.0
Team C,3,26.0
Team D,4,24.0
```

After running the simulation, the script will print the winning probabilities for each team and a tabular representation of the round winners.
License

## License
MIT
