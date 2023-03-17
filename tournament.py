import argparse
import math
from collections import defaultdict, Counter
from typing import List, Dict

import numpy as np
from scipy.optimize import minimize_scalar

from team import Team

# Take the difference in seeds between two teams; historically, these are the rate at which the higher seed wins.
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
        self.temperature = temperature or self.find_temperature_using_least_squares()
        self.adj_matrix = self.calculate_adj_matrix()
        self.noise = noise
        self.verbose = verbose

    def calculate_adj_matrix(self):
        num_teams = len(self.teams)
        adj_matrix = np.zeros((num_teams, num_teams))

        for i, team_i in enumerate(self.teams):
            for j, team_j in enumerate(self.teams[i + 1:], i + 1):
                p_win = self.calculate_win_probability(team_i, team_j)
                adj_matrix[i, j] = p_win
                adj_matrix[j, i] = 1 - p_win

        return adj_matrix

    def print_verbose(self, *args):
        if self.verbose:
            print(*args)

    def run_once(self):
        assert len(self.teams) > 0, "No teams in the tournament. Exiting."
        self.print_verbose(f"\nRound of {len(self.teams)}")
        self.print_verbose("teams in round: ", [f"{x.name} ({x.seed})" for x in self.teams])

        # Terminal recursive condition
        if len(self.teams) == 1:
            winner = self.teams[0]
            print(f"Winner: {winner.name}")
            return winner

        winners = []
        realized_upsets = 0
        matchups = [(self.teams[i], self.teams[i + 1]) for i in range(0, len(self.teams), 2)]

        for home, away in matchups:
            winner = self.select_winner_simple(home, away)
            loser = home if winner == away else away
            is_upset = winner.would_upset(loser)
            realized_upsets += int(is_upset)
            winners += [winner]

            if is_upset: self.print_upset(loser, winner)

        self.print_verbose(f"Upset rate for this round: {realized_upsets / len(winners):.2%}")

        # Recurse
        return Tournament(winners, self.noise, self.temperature, self.verbose).run_once()

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
        favorite = home if home_is_better else away
        underdog = away if home_is_better else home

        statistical = self.calculate_win_probability(underdog, favorite)

        # Noise is added by using historical upset rates rather than team specific KenPom scores
        probability = (1 - self.noise) * statistical + self.noise * historical_upset_rate

        # If a random number is less than the probability of an upset, return the underdog
        return underdog if np.random.rand() < probability else favorite

    def print_upset(self, loser, winner):
        expected_edge = winner.is_better_kenpom(loser)
        self.print_verbose(f"{winner.name}({winner.seed}) "
                           f"over {loser.name}({loser.seed}) "
                           f"{'' if expected_edge else 'UNEXPECTED'}")

    def get_team_by_name(self, team_name: str):
        return next((team for team in self.teams if team.name == team_name), None)

    def calculate_win_probability(self, home: Team, away: Team):
        return self.logistic(home.metric - away.metric, self.temperature)

    def average_kenpom_differences(self):
        # Initialize a dictionary to store the sum of KenPom differences and counts for each seed difference
        kenpom_sums = defaultdict(float)
        count = defaultdict(int)

        # Loop through all possible matchups between teams
        for i, home in enumerate(self.teams):
            for away in self.teams[i + 1:]:
                seed_diff = abs(home.seed - away.seed)
                kenpom_diff = abs(home.metric - away.metric)

                # Update the sum of KenPom differences and counts for the seed difference
                kenpom_sums[seed_diff] += kenpom_diff
                count[seed_diff] += 1

        # Calculate the average KenPom difference for each seed difference
        return {seed_diff: kenpom_sums[seed_diff] / count[seed_diff] for seed_diff in kenpom_sums}

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
            probability = Tournament.logistic(difference, temperature)

            # Add the squared error between the calculated probability and historical probability
            error += (probability - historical_probability) ** 2

        return error

    @staticmethod
    def logistic(x, k=1):
        return 1 / (1 + math.exp(-1 * k * x))

    @classmethod
    def simulate(cls, teams: List[Team], noise: float, num_iterations: int, **__):
        """
        Simulate the tournament multiple times to generate power scores for each team based on their
        winning frequencies. A final simulation is run with these power scores to determine the winner.

        We run the N initial simulations by sampling from a probability distribution constructed with kenpom scores.
        We when back-feed the simulated results into a final run. This captures the graph structure of the tournament,
        because the simulated win rate is impacted by where a given team exists in the bracket

        The back-feed step is what gives us an edge over sampling kenpom.

        :param teams: A list of Team objects participating in the tournament.
        :param noise: A float between 0 and 1 representing the noise level. Higher noise yields more upsets.
        :param num_iterations: The number of iterations to perform for the initial simulations.
        """

        # Run N simulations using the initial input metrics, and get the resulting win frequencies
        def get_winner():
            winning_team = Tournament(teams, noise).run_once()
            return winning_team.name

        win_counts = Counter(get_winner() for _ in range(num_iterations))
        frequencies = {team_name: win_count / num_iterations for team_name, win_count in win_counts.items()}

        # Calculate the power scores
        power_scores_ = cls.calculate_power_scores(frequencies)

        # print power scores sorted by value descending
        sorted_power_scores = sorted(power_scores_.items(), key=lambda x: x[1], reverse=True)
        print(sorted_power_scores)

        # Create the new teams array with power scores replacing the KenPom scores
        # This is the magic.
        teams = [Team(team.name, team.seed, power_scores_[team.name]) for team in teams]

        # Run one final simulation with the new power scores
        top_4_power_scores = {team_name for team_name, _ in sorted_power_scores[:4]}

        winner = None
        while not winner or winner.name not in top_4_power_scores:
            if winner:
                print(f"Re-running because {winner.name} is not in the top 4 power scores")
            final_tournament = Tournament(teams, noise, verbose=True)
            winner = final_tournament.run_once()

        print(f"\nFinal Winner: {winner.name}")

    @classmethod
    def calculate_power_scores(cls, win_frequencies: Dict[str, float]) -> defaultdict:
        min_freq = min(win_frequencies.values())
        max_freq = max(win_frequencies.values())

        def scale(x):
            return (x - min_freq) / (max_freq - min_freq)

        power_scores = defaultdict(float)
        power_scores.update({team: scale(freq) for team, freq in win_frequencies.items()})

        return power_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NCAA Tournament Simulation"
    )
    parser.add_argument(
        '-f', '--file',
        default='2023ncaab.csv',
        help="Path to the data file."
    )
    parser.add_argument(
        '-z', '--noise',
        type=float,
        default=0.5,
        help="Noise level from 0 to 1. Higher noise yields more upsets."
    )
    parser.add_argument(
        '-n', '--num_iterations',
        type=int,
        default=100,
        help="The number of simulated iterations to perform."
    )
    kwargs_ = vars(parser.parse_args())
    teams_ = Team.extract_teams(**kwargs_)
    Tournament.simulate(teams_, **kwargs_)
