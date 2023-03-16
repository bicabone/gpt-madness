from collections import defaultdict, Counter
from typing import List

import numpy as np
from numpy.ma import exp
from scipy.optimize import minimize_scalar
from scipy.special import expit

from ncaa.round_winners import RoundWinners
from ncaa.team_metric import TeamMetric

HISTORICAL_SEED_WIN_RATES = {
    (1, 16): 1.000,
    (2, 15): 0.917,
    (3, 14): 0.833,
    (4, 13): 0.750,
    (5, 12): 0.667,
    (6, 11): 0.583,
    (7, 10): 0.583,
    (8, 9): 0.500,
}


class Tournament:
    def __init__(self, team_metrics: List[TeamMetric]):
        self.team_metrics: List[TeamMetric] = team_metrics
        self.teams = [team_metric.team_name for team_metric in team_metrics]
        self.k = Tournament.find_best_k()
        self.adj_matrix = self.calculate_adj_matrix()
        self.round_win_counts = [Counter() for _ in range(int(np.log2(len(self.teams))))]
        self.round_winners = defaultdict(list)

    def calculate_adj_matrix(self):
        num_teams = len(self.team_metrics)
        adj_matrix = np.zeros((num_teams, num_teams))

        for i, team_i in enumerate(self.team_metrics):
            for j, team_j in enumerate(self.team_metrics):
                if i != j:
                    p_win = self.calculate_win_probability(team_i, team_j)
                    adj_matrix[i, j] = p_win
                    adj_matrix[j, i] = 1 - p_win

        return adj_matrix

    @staticmethod
    def error_function(k, average_kenpom_difference):
        error = 0
        for matchup, historical_probability in HISTORICAL_SEED_WIN_RATES.items():
            difference = average_kenpom_difference[matchup]
            probability = 1 / (1 + exp(-k * difference))
            error += (probability - historical_probability) ** 2
        return error

    @staticmethod
    def average_kenpom_difference(max_seed=16, kenpom_range=(0, 40)):
        min_kenpom, max_kenpom = kenpom_range
        kenpom_increment = (max_kenpom - min_kenpom) / max_seed
        average_difference = {}

        for higher_seed in range(1, max_seed + 1):
            for lower_seed in range(higher_seed + 1, max_seed + 1):
                higher_seed_kenpom = max_kenpom - (higher_seed - 1) * kenpom_increment
                lower_seed_kenpom = max_kenpom - (lower_seed - 1) * kenpom_increment
                average_difference[(higher_seed, lower_seed)] = higher_seed_kenpom - lower_seed_kenpom

        return average_difference

    @staticmethod
    def find_best_k():
        average_difference = Tournament.average_kenpom_difference()
        result = minimize_scalar(Tournament.error_function, args=(average_difference,))
        return result.x

    def calculate_win_probability(self, team_i: TeamMetric, team_j: TeamMetric):
        seed_diff = team_j.team_seed - team_i.team_seed
        ken_pom_diff = team_i.ken_pom_score - team_j.ken_pom_score
        return expit(self.k * (ken_pom_diff + seed_diff))

    def play_rounds(self):
        remaining_teams = list(range(len(self.teams)))
        round_num = 0

        while len(remaining_teams) > 1:
            winners = []
            for i in range(0, len(remaining_teams), 2):
                team_i = remaining_teams[i]
                team_j = remaining_teams[i + 1]
                p_win_i = self.adj_matrix[team_i, team_j]
                win_i = np.random.rand() < p_win_i
                winning_team_index = i if win_i else i + 1
                winners.append(winning_team_index)

            self.round_winners[round_num] = [self.teams[i] for i in winners]
            remaining_teams = winners
            round_num += 1

    def get_team_index_by_name(self, team_name):
        try:
            return self.teams.index(team_name)
        except ValueError:
            raise Exception(f"Team '{team_name}' not found in the teams list.")

    def calculate_round_win_averages(self, num_simulations):
        round_win_averages = [{} for _ in range(len(self.round_win_counts))]
        for i, round_win_count in enumerate(self.round_win_counts):
            for team, count in round_win_count.items():
                round_win_averages[i][team] = count / num_simulations
        return round_win_averages

    def run_simulations(self, num_simulations):
        simulation_results = defaultdict(int)

        for _ in range(num_simulations):
            self.round_winners.clear()  # Clear the round_winners dictionary before each simulation
            self.play_rounds()
            final_round_key = max(self.round_winners.keys())
            final_round_winners = self.round_winners[final_round_key]
            winning_team_name = final_round_winners[0]
            winner = self.teams[self.get_team_index_by_name(winning_team_name)]
            simulation_results[winner] += 1

        # Convert to probability
        for team in simulation_results:
            simulation_results[team] /= num_simulations

        round_win_averages = self.calculate_round_win_averages(num_simulations)
        round_winners = RoundWinners(self.round_winners)
        return simulation_results, round_winners, round_win_averages
