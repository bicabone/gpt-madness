from prettytable import PrettyTable


class RoundWinners:
    def __init__(self, round_winners_data):
        self.round_winners_data = round_winners_data
        self.rounds = len(round_winners_data)

    def print_tabular(self):
        table = PrettyTable()
        table.field_names = [f"Round {i + 1}" for i in range(self.rounds)]

        max_winners = max(len(winners) for winners in self.round_winners_data.values())
        for i in range(max_winners):
            row = [
                winners_[i] if i < len(winners_) else ""
                for winners_ in self.round_winners_data.values()
            ]
            table.add_row(row)

        print(table)
