import argparse
import math
from collections import defaultdict
from typing import List, Dict

import numpy as np
from numpy.ma import exp
from scipy.optimize import minimize_scalar

from team import Team

HISTORICAL_SEED_WIN_RATES = {
    15: 1.000,
    14: 0.9585,
    13: 0.917,
    12: 0.875,
    11: 0.833,
    10: 0.7915,
    9: 0.750,
    8: 0.7085,
    7: 0.667,
    6: 0.625,
    5: 0.583,
    4: 0.5615,
    3: 0.54,
    2: 0.52,
    1: 0.500,
    0: 0.500,
}


class Tournament:

    def __init__(self, teams: List[Team], noise: float, temperature: float = None, verbose: bool = False):
        self.teams: List[Team] = teams
        self.temperature = temperature if temperature is not None else self.find_temperature_using_least_squares()
        self.adj_matrix = self.calculate_adj_matrix()
        self.noise = noise
        self.verbose = verbose

    @staticmethod
    def get_opponent_index(team_index):
        return team_index + 1 if team_index % 2 == 0 else team_index - 1

    def calculate_adj_matrix(self):
        num_teams = len(self.teams)
        adj_matrix = np.zeros((num_teams, num_teams))

        for i, team_i in enumerate(self.teams):
            for j, team_j in enumerate(self.teams):
                if i != j:
                    p_win = self.calculate_win_probability(team_i, team_j)
                    adj_matrix[i, j] = p_win
                    adj_matrix[j, i] = 1 - p_win

        return adj_matrix

    def print_verbose(self, *args):
        if self.verbose:
            print(*args)

    def run(self):
        self.print_verbose(f"\nRound of {len(self.teams)}")
        self.print_verbose("teams in round: ", [
            f"{x.name} ({x.seed})"
            for x in self.teams
        ])
        if len(self.teams) == 0:
            self.print_verbose("No teams in the tournament. Exiting.")
            return None

        if len(self.teams) == 1:
            winner = self.teams[0]
            print(f"Winner: {winner.name}")
            return winner

        winners = self.play_round()
        updated_tournament = Tournament(winners, self.noise, self.temperature, self.verbose)
        return updated_tournament.run()

    @staticmethod
    def historical_upset_rate(seed1, seed2):
        return 1 - Tournament.get_midpoint_win_rate(seed1, seed2)

    @staticmethod
    def get_midpoint_win_rate(seed1, seed2):
        lower, higher = sorted((seed1, seed2))
        return HISTORICAL_SEED_WIN_RATES.get(higher - lower)

    def select_winner_simple(self, home: Team, away: Team):
        home_seed, away_seed = home.seed, away.seed
        historical_upset_rate = self.historical_upset_rate(home_seed, away_seed)
        home_is_better = home.would_upset(away)
        better_team = home if home_is_better else away
        worse_team = away if home_is_better else home

        statistical = self.calculate_win_probability(worse_team, better_team)

        # Noise is added by using historical upset rates rather than team specific KenPom scores
        probability = (1 - self.noise) * statistical + self.noise * historical_upset_rate

        # If a random number is less than the probability of an upset, return the underdog
        if np.random.rand() < probability:
            return worse_team
        # Otherwise, return the favorite
        else:
            return better_team

    def play_round(self):
        winners = []
        realized_upsets = 0

        for i in range(0, len(self.teams), 2):
            home = self.teams[i]
            away = self.teams[i + 1]
            winner = self.select_winner_simple(home, away)
            loser = home if winner == away else away
            is_upset = winner.would_upset(loser)
            realized_upsets += 1 if is_upset else 0
            winners += [winner]

            if is_upset:
                expected_edge = winner.is_better_kenpom(loser)
                self.print_verbose(f"{winner.name}({winner.seed}) "
                                   f"over {loser.name}({loser.seed}) "
                                   f"{'' if expected_edge else 'UNEXPECTED'}")

        self.print_verbose(f"Upset rate for this round: {realized_upsets / len(winners):.2%}")

        return winners

    def get_team_by_name(self, team_name: str):
        for team in self.teams:
            if team.name == team_name:
                return team
        return None

    def calculate_win_probability(self, team_i: Team, team_j: Team):
        ken_pom_diff = team_i.metric - team_j.metric
        probability = 1 / (1 + math.exp(-self.temperature * ken_pom_diff))
        return probability

    def average_kenpom_differences(self):
        # Initialize a dictionary to store the sum of KenPom differences and counts for each seed difference
        kenpom_diff_sum = defaultdict(float)
        kenpom_diff_count = defaultdict(int)

        # Loop through all possible matchups between teams
        for i, home in enumerate(self.teams):
            for away in self.teams[i + 1:]:
                seed_diff = abs(home.seed - away.seed)
                kenpom_diff = abs(home.metric - away.metric)

                # Update the sum of KenPom differences and counts for the seed difference
                kenpom_diff_sum[seed_diff] += kenpom_diff
                kenpom_diff_count[seed_diff] += 1

        # Calculate the average KenPom difference for each seed difference
        average_difference = {}
        for seed_diff in kenpom_diff_sum:
            average_difference[seed_diff] = kenpom_diff_sum[seed_diff] / kenpom_diff_count[seed_diff]

        return average_difference

    def find_temperature_using_least_squares(self):
        average_difference = self.average_kenpom_differences()
        result = minimize_scalar(Tournament.error_function, args=average_difference)
        return result.x

    @staticmethod
    def error_function(temperature, average_kenpom_differences):
        error = 0
        for seed_difference, historical_probability in HISTORICAL_SEED_WIN_RATES.items():
            # Get the historical probability based on the seed difference
            historical_probability = HISTORICAL_SEED_WIN_RATES[seed_difference]

            # Calculate the probability based on the KenPom difference and the given k
            difference = average_kenpom_differences[seed_difference]
            probability = 1 / (1 + exp(-temperature * difference))

            # Add the squared error between the calculated probability and historical probability
            error += (probability - historical_probability) ** 2

        return error


def run_multiple_tournaments(teams: List[Team], noise: float, num_iterations: int) -> Dict[str, int]:
    win_counts = defaultdict(int)
    for i in range(num_iterations):
        tournament = Tournament(teams, noise)
        winner = tournament.run()
        win_counts[winner.name] += 1

    return {
        team_name: win_count / num_iterations
        for team_name, win_count in win_counts.items()
    }


def calculate_power_scores(win_frequencies: Dict[str, float]) -> defaultdict:
    min_freq = min(win_frequencies.values())
    max_freq = max(win_frequencies.values())

    power_scores = defaultdict(lambda: 0.0)
    for team_name, freq in win_frequencies.items():
        power_scores[team_name] = (freq - min_freq) / (max_freq - min_freq)

    return power_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NCAA Tournament Simulation"
    )
    parser.add_argument(
        '-f', '--file',
        default='2023ncaab.csv',
        help="Path to the data file (default: '2023ncaab.csv')"
    )
    parser.add_argument(
        '-z', '--noise',
        type=int,
        default=0.5
    )
    parser.add_argument(
        '-n', '--num_iterations',
        type=int,
        default=10000,
        help="Number of iterations for the frequency calculation (default: 1000)"
    )
    args = parser.parse_args()
    teams_ = Team.extract_teams(args.file)

    # Calculate the win frequencies
    win_frequencies_ = run_multiple_tournaments(teams_, args.noise, args.num_iterations)

    # Calculate the power scores
    power_scores_ = calculate_power_scores(win_frequencies_)
    # print power scores sorted by value descending
    sorted_power_scores = sorted(power_scores_.items(), key=lambda x: x[1], reverse=True)
    print(sorted_power_scores)

    # Create the new teams array with power scores replacing the KenPom scores
    # This is the magic. We back-feed the simulated results into a final run.
    # This captures the graph structure of the tournament, because your power score win rate
    # is impacted by where you live in the bracket
    # This step is what makes the simulation different from just sampling kenpom
    teams_ = [Team(team.name, team.seed, power_scores_[team.name]) for team in teams_]

    # Run one final simulation with the new power scores
    top_4_power_scores = {team_name for team_name, _ in sorted_power_scores[:4]}

    while True:
        final_tournament = Tournament(teams_, args.noise, verbose=True)
        winner = final_tournament.run()
        if winner.name in top_4_power_scores:
            break
        else:
            print(f"Re-running because {winner.name} is not in the top 4 power scores")

    print(f"\nFinal Winner: {winner.name}")
