from unittest import TestCase

from gpt_madness import main
from ncaa.team_metric import TeamMetric
from ncaa.tournament import Tournament


class TestTournament(TestCase):
    def test_tournament(self):
        team_metrics = [
            TeamMetric("Team A", 1, 30),
            TeamMetric("Team B", 16, 5),
            TeamMetric("Team C", 8, 22),
            TeamMetric("Team D", 9, 20),
            # ... Add more teams
        ]

        # Create a tournament object with team metrics
        tournament = Tournament(team_metrics)

        # Run the simulations
        num_simulations = 1000
        simulation_results, round_winners = tournament.run_simulations(num_simulations)

        # Print the results
        print("Simulation Results:")
        for team, probability in simulation_results.items():
            print(f"{team}: {probability * 100:.2f}%")

        print("\nRound Winners:")
        round_winners.print_tabular()

    def test_main(self):
        main()
