# NCAA Basketball Tournament Simulation

This project is a command-line tool for simulating the NCAA basketball tournament results with KenPom scores, written via ChatGPT~4.

We make a probability distribution using kenpom scores, which we sample from for thousands of tournaments. We then use the frequencies from this simulation to create new power scores that now consider the graph structure of the tournament, and we do one final run. 

## Installation

1. Create and activate a virtual environment (optional):

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

`python3 tournament.py teams.csv`

The CSV file should have columns named `team`, `seed`, and `score`. Here's an example:

```csv
team,seed,score
Team A,1,30.0
Team B,16,28.0
Team C,2,26.0
Team D,15,24.0
```

Note: it is important that the csv is sorted in the order of the bracket itself
- SOUTH->EAST->MIDWEST->WEST
- 16->1->15->2->...->9->8

After running the simulation, the script will print the winning probabilities for each team and a tabular representation of the round winners.
License

## License
MIT
