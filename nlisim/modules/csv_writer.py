import csv
import itertools

import attr

from nlisim.module import ModuleModel, ModuleState
from nlisim.postprocess import generate_summary_stats
from nlisim.state import State


@attr.s(kw_only=True, repr=False)
class CSVWriterState(ModuleState):
    def __repr__(self):
        return 'CSVWriterState()'


class CSVWriter(ModuleModel):
    name = 'csv_writer'

    StateClass = CSVWriterState

    def advance(self, state: State, previous_time: float) -> State:
        now = state.time

        summary_stats = generate_summary_stats(state)
        data_columns = [now,] + list(
            itertools.chain.from_iterable(
                list(module_stats.values()) for module, module_stats in summary_stats.items()
            )
        )
        with open('data.csv', 'a') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(data_columns)

        return state

    def initialize(self, state: State) -> State:
        summary_stats = generate_summary_stats(state)
        column_names = ['time'] + list(
            itertools.chain.from_iterable(
                [module + '-' + statistic_name for statistic_name in module_stats.keys()]
                for module, module_stats in summary_stats.items()
            )
        )
        with open('data.csv', 'w') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(column_names)

        return state
