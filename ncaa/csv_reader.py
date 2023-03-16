import csv
import os

from ncaa.team_metric import TeamMetric


def read_csv_to_team_metrics(file_path: str):
    team_metrics = []

    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    full_file_path = os.path.join(parent_dir, file_path)

    with open(full_file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            team_name = row["team"]
            team_seed = int(row["seed"])
            ken_pom_score = float(row["score"])

            team_metric = TeamMetric(team_name, team_seed, ken_pom_score)
            team_metrics.append(team_metric)

        return team_metrics
