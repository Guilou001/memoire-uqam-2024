from typing import Union

import pandas as pd


class RebalanceScheduler:
    WEEKDAY_MAP = {
        'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3,
        'FRI': 4, 'SAT': 5, 'SUN': 6,
        0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU',
        4: 'FRI', 5: 'SAT', 6: 'SUN'
    }

    FREQ_MAPPING = {'M': 1, 'Q': 3, 'S': 6, 'A': 12}

    def __init__(self, end_date: str, first_rebalancing_year: int, rebalancing_month: int,
                 rebalancing_week: int, rebalancing_weekday: Union[str, int], rebalancing_frequency: str):
        self.end_date = pd.Timestamp(end_date)
        self.first_rebalancing_year = first_rebalancing_year
        self.rebalancing_month = rebalancing_month
        self.rebalancing_week = rebalancing_week
        self.rebalancing_weekday = self.map_weekday(rebalancing_weekday)  # Monday is 0, Sunday is 6
        self.rebalancing_frequency = rebalancing_frequency.upper()  # 'D', 'W', 'M', 'Q', 'S', 'A'
        self.initial_rebalancing_date = self.generate_initial_rebalancing_date()
        self.rebalancing_dates = self.generate_rebalancing_dates()
        self.next_rebalance = None
        self.previous_rebalance = None
        self.set_next_and_previous_rebalancing()

    @staticmethod
    def map_weekday(weekday: Union[int, str]) -> int:
        """
        Maps a weekday to its corresponding integer index.
        """
        if isinstance(weekday, str):
            return RebalanceScheduler.WEEKDAY_MAP[weekday.upper()]
        elif isinstance(weekday, int) and 0 <= weekday <= 6:
            return weekday
        else:
            raise ValueError("Invalid weekday format. Weekday must be a string or an integer between 0 and 6.")

    def generate_initial_rebalancing_date(self) -> pd.Timestamp:
        """
        Generates the initial rebalancing date based on the year, month, week, and weekday specified.
        """
        first_day_of_month = pd.Timestamp(year=self.first_rebalancing_year, month=self.rebalancing_month, day=1)
        first_occurrence = (self.rebalancing_weekday - first_day_of_month.weekday() + 7) % 7
        day_of_month = first_occurrence + 1 + (self.rebalancing_week - 1) * 7
        return pd.Timestamp(year=self.first_rebalancing_year, month=self.rebalancing_month, day=day_of_month)

    def generate_rebalancing_dates(self) -> list[pd.Timestamp]:
        """
        Generates rebalancing dates aligned with the initial_rebalance_date date,
        considering the specified frequency until the end_date.
        """
        if self.rebalancing_frequency == 'D':
            return pd.bdate_range(start=self.initial_rebalancing_date, end=self.end_date).tolist()
        elif self.rebalancing_frequency == 'W':
            return pd.date_range(start=self.initial_rebalancing_date, end=self.end_date,
                                 freq=f'W-{self.WEEKDAY_MAP[self.rebalancing_weekday]}').tolist()
        else:
            return self._generate_monthly_rebalancing_dates()

    def _generate_monthly_rebalancing_dates(self) -> list[pd.Timestamp]:
        """
        Helper function to generate rebalancing dates for frequencies 'M', 'Q', 'S', 'A'.
        """
        freq_in_months = self.FREQ_MAPPING[self.rebalancing_frequency]
        rebalance_dates = [self.initial_rebalancing_date]
        next_date = self.initial_rebalancing_date

        while next_date <= self.end_date:
            next_date = self._add_months(next_date, freq_in_months)
            if next_date <= self.end_date:
                rebalance_dates.append(next_date)

        return rebalance_dates

    def _add_months(self, date: pd.Timestamp, months: int) -> pd.Timestamp:
        """
        Adds a specified number of months to a date while aligning to the same week and weekday of the month.
        """
        target_date = date + pd.DateOffset(months=months)
        first_day_of_next_month = pd.Timestamp(year=target_date.year, month=target_date.month, day=1)
        first_occurrence = (self.rebalancing_weekday - first_day_of_next_month.weekday() + 7) % 7
        day_of_month = first_occurrence + 1 + (self.rebalancing_week - 1) * 7

        while True:
            try:
                return pd.Timestamp(year=target_date.year, month=target_date.month, day=day_of_month)
            except ValueError:
                day_of_month -= 1

    def set_next_and_previous_rebalancing(self):
        """
        Sets the next and previous rebalancing dates dynamically based on current date.
        """
        now = pd.Timestamp.now()
        future_dates = [date for date in self.rebalancing_dates if date > now]
        past_dates = [date for date in self.rebalancing_dates if date <= now]

        self.next_rebalance = future_dates[0] if future_dates else None
        self.previous_rebalance = past_dates[-1] if past_dates else None

    def get_next_rebalancing(self) -> Union[pd.Timestamp, None]:
        """
        Returns the next rebalancing date.
        """
        return self.next_rebalance

    def get_previous_rebalancing(self) -> Union[pd.Timestamp, None]:
        """
        Returns the previous rebalancing date.
        """
        return self.previous_rebalance

    def __repr__(self) -> str:
        """
        Provides a string representation of the object.
        """
        return (f"RebalanceScheduler(end_date={self.end_date}, initial_rebalance_date={self.initial_rebalancing_date}, "
                f"rebalance_frequency={self.rebalancing_frequency}, next_rebalance={self.next_rebalance}, "
                f"previous_rebalance={self.previous_rebalance})")


if __name__ == "__main__":
    pass



