from abc import ABC, abstractmethod

import pandas as pd

from ml_returns_pred.file_management.file_manager import FileManagerStatic
from ml_returns_pred.file_management.merge_data_from_folder import BatchMergeDataFromFolders
from ml_returns_pred.read_config.config_reader import get_merged_config
from ml_returns_pred.visualization.strategy_returns_viz import StrategyReturnsViz


class AbstractResultsAggregator(ABC):

    aggregation_dict_path: str = "../config/meta_config/aggregate_results.yaml"

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def aggregate_data(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def visualize_strategy_returns(self, **kwargs) -> None:
        pass

    @abstractmethod
    def main(self) -> None:
        pass

    @staticmethod
    def print_separator(message: str) -> None:
        separator_line = "█" * 2 + "=" * 36 + "★★★" + "=" * 36 + "█" * 2
        total_width = len(separator_line)
        title_line_1 = "█" + " " * (total_width - 2) + "█"

        # Calculate the amount of padding needed on each side of the message
        message_length = len(message)
        padding_each_side = (total_width - 2 - message_length) // 2
        extra_padding = (total_width - 2 - message_length) % 2  # In case the message length is odd

        title_line_2 = "█" + " " * padding_each_side + message + " " * (padding_each_side + extra_padding) + "█"
        title_line_3 = title_line_1

        print(f"\n{separator_line}\n{title_line_1}\n{title_line_2}\n{title_line_3}\n{separator_line}\n")


class ResultsAggregator(AbstractResultsAggregator):

    def __init__(self):
        strategy_name = "results_aggregator"
        config_path = [self.aggregation_dict_path]
        config = get_merged_config(config_paths=config_path, strategy_name=strategy_name)
        super().__init__(config)

    def aggregate_data(self) -> list[pd.DataFrame]:
        aggregated_data_dict = self.config["aggregate_data"]

        merged_data = BatchMergeDataFromFolders(
            folder_paths=aggregated_data_dict["folder_paths"],
            save_paths=aggregated_data_dict["save_paths"],
        )

        merged_data = merged_data.batch_merge(
            read_kwargs=aggregated_data_dict["read_kwargs"],
            merge_kwargs=aggregated_data_dict["merge_kwargs"],
            rename_strategy=aggregated_data_dict["rename_strategy"],
            save_data=aggregated_data_dict["save_data"],
            save_kwargs=aggregated_data_dict["save_kwargs"],
        )

        self.print_separator(message="Data aggregated successfully!")

        return merged_data

    def visualize_strategy_returns(self) -> None:
        visualization_dict = self.config["visualize_data"]

        benchmark = FileManagerStatic().load_data(
            relative_file_path=visualization_dict["benchmark_path"],
            **visualization_dict["benchmark_read_kwargs"]
        ).squeeze()

        aggregated_strategy_returns = FileManagerStatic().load_data(
            relative_file_path=visualization_dict["aggregated_strategy_returns_path"],
            **visualization_dict["aggregated_strategy_returns_read_kwargs"]
        )

        strategy_returns_viz = StrategyReturnsViz(
            df=aggregated_strategy_returns,
            benchmark_returns_paths=visualization_dict['benchmark_returns_paths'],
        )

        strategy_returns_viz.plot_cumulative_returns(
            method=visualization_dict["method"],
            title=visualization_dict["title"],
            ylabel=visualization_dict["ylabel"],
            xlabel=visualization_dict["xlabel"],
            save_path=visualization_dict["save_path"],
            subplots_kwargs=visualization_dict["subplot_kwargs"],
            log_scale=visualization_dict["log_scale"],
        )

        self.print_separator(message="Data visualized successfully!")

    def main(self) -> None:
        merged_data = self.aggregate_data()
        self.visualize_strategy_returns()



