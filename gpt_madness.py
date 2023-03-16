import sys

from ncaa.csv_reader import read_csv_to_team_metrics
from ncaa.tournament import Tournament


def main(file_path: str="2023ncaab.csv"):
    team_metrics = read_csv_to_team_metrics(file_path)
    tournament = Tournament(team_metrics)

    # Run the tournament simulations
    num_simulations = 100000
    winning_probabilities, round_winners, averages = tournament.run_simulations(num_simulations)

    # Print winning probabilities
    for team, probability in winning_probabilities.items():
        print(f"{team}: Winning Probability: {probability:.4f}")

    # Print round winners in a tabular format
    round_winners.print_tabular()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <csv_file_path>")
        sys.exit(1)

    main(sys.argv[1])
