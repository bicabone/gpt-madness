import csv
import os

import dataclasses

PARENT_DIR = os.path.dirname(os.path.realpath(__file__))


@dataclasses.dataclass
class Team:
    name: str
    seed: int
    metric: float

    def would_upset(self, other_team):
        return self.seed < other_team.seed
    def is_better_kenpom(self, other_team):
        return self.metric > other_team.metric

    @classmethod
    def extract_teams(cls, file_path: str):
        with open(os.path.join(PARENT_DIR, file_path), "r", newline="") as csvfile:
            return [cls(
                name=row["team"],
                seed=int(row["seed"]),
                metric=float(row["score"])
            ) for row in (csv.DictReader(csvfile))]
