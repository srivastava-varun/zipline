#
# Copyright 2013 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

Risk Report
===========

    +-----------------+----------------------------------------------------+
    | key             | value                                              |
    +=================+====================================================+
    | trading_days    | The number of trading days between self.start_date |
    |                 | and self.end_date                                  |
    +-----------------+----------------------------------------------------+
    | benchmark_volat\| The volatility of the benchmark between            |
    | ility           | self.start_date and self.end_date.                 |
    +-----------------+----------------------------------------------------+
    | algo_volatility | The volatility of the algo between self.start_date |
    |                 | and self.end_date.                                 |
    +-----------------+----------------------------------------------------+
    | treasury_period\| The return of treasuries over the period. Treasury |
    | _return         | maturity is chosen to match the duration of the    |
    |                 | test period.                                       |
    +-----------------+----------------------------------------------------+
    | sharpe          | The sharpe ratio based on the _algorithm_ (rather  |
    |                 | than the static portfolio) returns.                |
    +-----------------+----------------------------------------------------+
    | information     | The information ratio based on the _algorithm_     |
    |                 | (rather than the static portfolio) returns.        |
    +-----------------+----------------------------------------------------+
    | beta            | The _algorithm_ beta to the benchmark.             |
    +-----------------+----------------------------------------------------+
    | alpha           | The _algorithm_ alpha to the benchmark.            |
    +-----------------+----------------------------------------------------+
    | excess_return   | The excess return of the algorithm over the        |
    |                 | treasuries.                                        |
    +-----------------+----------------------------------------------------+
    | max_drawdown    | The largest relative peak to relative trough move  |
    |                 | for the portfolio returns between self.start_date  |
    |                 | and self.end_date.                                 |
    +-----------------+----------------------------------------------------+


"""

import logbook
import datetime
import math
import numpy as np
import numpy.linalg as la
from dateutil.relativedelta import relativedelta

import zipline.finance.trading as trading
from zipline.utils.date_utils import epoch_now
import zipline.utils.math_utils as zp_math

import pandas as pd

log = logbook.Logger('Risk')


TREASURY_DURATIONS = [
    '1month', '3month', '6month',
    '1year', '2year', '3year', '5year',
    '7year', '10year', '30year'
]


# check if a field in rval is nan, and replace it with
# None.
def check_entry(key, value):
    if key != 'period_label':
        return np.isnan(value) or np.isinf(value)
    else:
        return False


############################
# Risk Metric Calculations #
############################


def sharpe_ratio(algorithm_volatility, algorithm_return, treasury_return):
    """
    http://en.wikipedia.org/wiki/Sharpe_ratio

    Args:
        algorithm_volatility (float): Algorithm volatility.
        algorithm_return (float): Algorithm return percentage.
        treasury_return (float): Treasury return percentage.

    Returns:
        float. The Sharpe ratio.
    """
    if zp_math.tolerant_equals(algorithm_volatility, 0):
        return 0.0

    return (algorithm_return - treasury_return) / algorithm_volatility


def sortino_ratio(algorithm_returns, algorithm_period_return, mar):
    """
    http://en.wikipedia.org/wiki/Sortino_ratio

    Args:
        algorithm_returns (np.array-like):
            Returns from algorithm lifetime.
        algorithm_period_return (float):
            Algorithm return percentage from latest period.
        mar (float): Minimum acceptable return.

    Returns:
        float. The Sortino ratio.
    """
    if len(algorithm_returns) == 0:
        return 0.0

    rets = algorithm_returns
    downside = (rets[rets < mar] - mar) ** 2
    dr = np.sqrt(downside.sum() / len(rets))

    if zp_math.tolerant_equals(dr, 0):
        return 0.0

    return (algorithm_period_return - mar) / dr


def information_ratio(algorithm_returns, benchmark_returns):
    """
    http://en.wikipedia.org/wiki/Information_ratio

    Args:
        algorithm_returns (np.array-like):
            All returns during algorithm lifetime.
        benchmark_returns (np.array-like):
            All benchmark returns during algo lifetime.

    Returns:
        float. Information ratio.
    """
    relative_returns = algorithm_returns - benchmark_returns

    relative_deviation = relative_returns.std(ddof=1)

    if (
        zp_math.tolerant_equals(relative_deviation, 0)
        or
        np.isnan(relative_deviation)
    ):
        return 0.0

    return np.mean(relative_returns) / relative_deviation


def alpha(algorithm_period_return, treasury_period_return,
          benchmark_period_returns, beta):
    """
    http://en.wikipedia.org/wiki/Alpha_(investment)

    Args:
        algorithm_period_return (float):
            Return percentage from algorithm period.
        treasury_period_return (float):
            Return percentage for treasury period.
        benchmark_period_return (float):
            Return percentage for benchmark period.
        beta (float):
            beta value for the same period as all other values

    Returns:
        float. The alpha of the algorithm.
    """
    return algorithm_period_return - \
        (treasury_period_return + beta *
         (benchmark_period_returns - treasury_period_return))

###########################
# End Risk Metric Section #
###########################


def get_treasury_rate(treasury_curves, treasury_duration, day):
    rate = None

    curve = treasury_curves[day]
    # 1month note data begins in 8/2001,
    # so we can use 3month instead.
    idx = TREASURY_DURATIONS.index(treasury_duration)
    for duration in TREASURY_DURATIONS[idx:]:
        rate = curve[duration]
        if rate is not None:
            break

    return rate


def search_day_distance(end_date, dt):
    tdd = trading.environment.trading_day_distance(dt, end_date)
    if tdd is None:
        return None
    assert tdd >= 0
    return tdd


def select_treasury_duration(start_date, end_date):
    td = end_date - start_date
    if td.days <= 31:
        treasury_duration = '1month'
    elif td.days <= 93:
        treasury_duration = '3month'
    elif td.days <= 186:
        treasury_duration = '6month'
    elif td.days <= 366:
        treasury_duration = '1year'
    elif td.days <= 365 * 2 + 1:
        treasury_duration = '2year'
    elif td.days <= 365 * 3 + 1:
        treasury_duration = '3year'
    elif td.days <= 365 * 5 + 2:
        treasury_duration = '5year'
    elif td.days <= 365 * 7 + 2:
        treasury_duration = '7year'
    elif td.days <= 365 * 10 + 2:
        treasury_duration = '10year'
    else:
        treasury_duration = '30year'

    return treasury_duration


def choose_treasury(treasury_curves, start_date, end_date):
    treasury_duration = select_treasury_duration(start_date, end_date)
    end_day = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    search_day = None

    if end_day in treasury_curves:
        rate = get_treasury_rate(treasury_curves,
                                 treasury_duration,
                                 end_day)
        if rate is not None:
            search_day = end_day

    if not search_day:
        # in case end date is not a trading day or there is no treasury
        # data, search for the previous day with an interest rate.
        search_days = treasury_curves.index

        # Find rightmost value less than or equal to end_day
        i = search_days.searchsorted(end_day)
        for prev_day in search_days[i - 1::-1]:
            rate = get_treasury_rate(treasury_curves,
                                     treasury_duration,
                                     prev_day)
            if rate is not None:
                search_day = prev_day
                search_dist = search_day_distance(end_date, prev_day)
                break

        if search_day:
            if (search_dist is None or search_dist > 1) and \
                    search_days[0] <= end_day <= search_days[-1]:
                message = "No rate within 1 trading day of end date = \
{dt} and term = {term}. Using {search_day}. Check that date doesn't exceed \
treasury history range."
                message = message.format(dt=end_date,
                                         term=treasury_duration,
                                         search_day=search_day)
                log.warn(message)

    if search_day:
        td = end_date - start_date
        return rate * (td.days + 1) / 365

    message = "No rate for end date = {dt} and term = {term}. Check \
that date doesn't exceed treasury history range."
    message = message.format(
        dt=end_date,
        term=treasury_duration
    )
    raise Exception(message)


class RiskMetricsPeriod(object):
    def __init__(self, start_date, end_date, returns,
                 benchmark_returns=None):

        treasury_curves = trading.environment.treasury_curves
        if treasury_curves.index[-1] >= start_date:
            mask = ((treasury_curves.index >= start_date) &
                    (treasury_curves.index <= end_date))

            self.treasury_curves = treasury_curves[mask]
        else:
            # our test is beyond the treasury curve history
            # so we'll use the last available treasury curve
            self.treasury_curves = treasury_curves[-1:]

        self.start_date = start_date
        self.end_date = end_date

        if benchmark_returns is None:
            benchmark_returns = [
                x for x in trading.environment.benchmark_returns
                if x.date >= returns[0].date and
                x.date <= returns[-1].date
            ]

        self.algorithm_returns = self.mask_returns_to_period(returns)
        self.benchmark_returns = self.mask_returns_to_period(benchmark_returns)
        self.calculate_metrics()

    def calculate_metrics(self):

        self.benchmark_period_returns = \
            self.calculate_period_returns(self.benchmark_returns)

        self.algorithm_period_returns = \
            self.calculate_period_returns(self.algorithm_returns)

        if not self.algorithm_returns.index.equals(
            self.benchmark_returns.index
        ):
            message = "Mismatch between benchmark_returns ({bm_count}) and \
            algorithm_returns ({algo_count}) in range {start} : {end}"
            message = message.format(
                bm_count=len(self.benchmark_returns),
                algo_count=len(self.algorithm_returns),
                start=self.start_date,
                end=self.end_date
            )
            raise Exception(message)

        self.num_trading_days = len(self.benchmark_returns)
        self.benchmark_volatility = self.calculate_volatility(
            self.benchmark_returns)
        self.algorithm_volatility = self.calculate_volatility(
            self.algorithm_returns)
        self.treasury_period_return = choose_treasury(
            self.treasury_curves,
            self.start_date,
            self.end_date
        )
        self.sharpe = self.calculate_sharpe()
        self.sortino = self.calculate_sortino()
        self.information = self.calculate_information()
        self.beta, self.algorithm_covariance, self.benchmark_variance, \
            self.condition_number, self.eigen_values = self.calculate_beta()
        self.alpha = self.calculate_alpha()
        self.excess_return = self.algorithm_period_returns - \
            self.treasury_period_return
        self.max_drawdown = self.calculate_max_drawdown()

    def to_dict(self):
        """
        Creates a dictionary representing the state of the risk report.
        Returns a dict object of the form:
        """
        period_label = self.end_date.strftime("%Y-%m")
        rval = {
            'trading_days': self.num_trading_days,
            'benchmark_volatility': self.benchmark_volatility,
            'algo_volatility': self.algorithm_volatility,
            'treasury_period_return': self.treasury_period_return,
            'algorithm_period_return': self.algorithm_period_returns,
            'benchmark_period_return': self.benchmark_period_returns,
            'sharpe': self.sharpe,
            'sortino': self.sortino,
            'information': self.information,
            'beta': self.beta,
            'alpha': self.alpha,
            'excess_return': self.excess_return,
            'max_drawdown': self.max_drawdown,
            'period_label': period_label
        }

        return {k: None if check_entry(k, v) else v
                for k, v in rval.iteritems()}

    def __repr__(self):
        statements = []
        metrics = [
            "algorithm_period_returns",
            "benchmark_period_returns",
            "excess_return",
            "num_trading_days",
            "benchmark_volatility",
            "algorithm_volatility",
            "sharpe",
            "sortino",
            "information",
            "algorithm_covariance",
            "benchmark_variance",
            "beta",
            "alpha",
            "max_drawdown",
            "algorithm_returns",
            "benchmark_returns",
            "condition_number",
            "eigen_values"
        ]

        for metric in metrics:
            value = getattr(self, metric)
            statements.append("{m}:{v}".format(m=metric, v=value))

        return '\n'.join(statements)

    def mask_returns_to_period(self, daily_returns):
        if isinstance(daily_returns, list):
            returns = pd.Series([x.returns for x in daily_returns],
                                index=[x.date for x in daily_returns])
        else:  # otherwise we're receiving an index already
            returns = daily_returns

        trade_days = trading.environment.trading_days
        trade_day_mask = returns.index.normalize().isin(trade_days)

        mask = ((returns.index >= self.start_date) &
                (returns.index <= self.end_date) & trade_day_mask)

        returns = returns[mask]
        return returns

    def calculate_period_returns(self, returns):
        period_returns = (1. + returns).prod() - 1
        return period_returns

    def calculate_volatility(self, daily_returns):
        return np.std(daily_returns, ddof=1) * math.sqrt(self.num_trading_days)

    def calculate_sharpe(self):
        """
        http://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return sharpe_ratio(self.algorithm_volatility,
                            self.algorithm_period_returns,
                            self.treasury_period_return)

    def calculate_sortino(self, mar=None):
        """
        http://en.wikipedia.org/wiki/Sortino_ratio
        """
        if mar is None:
            mar = self.treasury_period_return

        return sortino_ratio(self.algorithm_returns,
                             self.algorithm_period_returns,
                             mar)

    def calculate_information(self):
        """
        http://en.wikipedia.org/wiki/Information_ratio
        """
        return information_ratio(self.algorithm_returns,
                                 self.benchmark_returns)

    def calculate_beta(self):
        """

        .. math::

            \\beta_a = \\frac{\mathrm{Cov}(r_a,r_p)}{\mathrm{Var}(r_p)}

        http://en.wikipedia.org/wiki/Beta_(finance)
        """
        #it doesn't make much sense to calculate beta for less than two days,
        #so return none.
        if len(self.algorithm_returns) < 2:
            return 0.0, 0.0, 0.0, 0.0, []

        returns_matrix = np.vstack([self.algorithm_returns,
                                    self.benchmark_returns])
        C = np.cov(returns_matrix, ddof=1)
        eigen_values = la.eigvals(C)
        condition_number = max(eigen_values) / min(eigen_values)
        algorithm_covariance = C[0][1]
        benchmark_variance = C[1][1]
        beta = algorithm_covariance / benchmark_variance

        return (
            beta,
            algorithm_covariance,
            benchmark_variance,
            condition_number,
            eigen_values
        )

    def calculate_alpha(self):
        """
        http://en.wikipedia.org/wiki/Alpha_(investment)
        """
        return alpha(self.algorithm_period_returns,
                     self.treasury_period_return,
                     self.benchmark_period_returns,
                     self.beta)

    def calculate_max_drawdown(self):
        compounded_returns = []
        cur_return = 0.0
        for r in self.algorithm_returns:
            try:
                cur_return += math.log(1.0 + r)
            #this is a guard for a single day returning -100%
            except ValueError:
                log.debug("{cur} return, zeroing the returns".format(
                    cur=cur_return))
                cur_return = 0.0
                # BUG? Shouldn't this be set to log(1.0 + 0) ?
            compounded_returns.append(cur_return)

        cur_max = None
        max_drawdown = None
        for cur in compounded_returns:
            if cur_max is None or cur > cur_max:
                cur_max = cur

            drawdown = (cur - cur_max)
            if max_drawdown is None or drawdown < max_drawdown:
                max_drawdown = drawdown

        if max_drawdown is None:
            return 0.0

        return 1.0 - math.exp(max_drawdown)


class RiskReport(object):
    def __init__(self, algorithm_returns, sim_params, benchmark_returns=None):
        """
        algorithm_returns needs to be a list of daily_return objects
        sorted in date ascending order
        """

        self.algorithm_returns = algorithm_returns
        self.sim_params = sim_params
        self.benchmark_returns = benchmark_returns
        self.created = epoch_now()

        if len(self.algorithm_returns) == 0:
            start_date = self.sim_params.period_start
            end_date = self.sim_params.period_end
        else:
            # FIXME: Papering over multiple algorithm_return types
            if isinstance(self.algorithm_returns, list):
                start_date = self.algorithm_returns[0].date
                end_date = self.algorithm_returns[-1].date
            else:
                start_date = self.algorithm_returns.index[0]
                end_date = self.algorithm_returns.index[-1]

        self.month_periods = self.periods_in_range(1, start_date, end_date)
        self.three_month_periods = self.periods_in_range(3, start_date,
                                                         end_date)
        self.six_month_periods = self.periods_in_range(6, start_date, end_date)
        self.year_periods = self.periods_in_range(12, start_date, end_date)

    def to_dict(self):
        """
        RiskMetrics are calculated for rolling windows in four lengths::
            - 1_month
            - 3_month
            - 6_month
            - 12_month

        The return value of this funciton is a dictionary keyed by the above
        list of durations. The value of each entry is a list of RiskMetric
        dicts of the same duration as denoted by the top_level key.

        See :py:meth:`RiskMetrics.to_dict` for the detailed list of fields
        provided for each period.
        """
        return {
            'one_month': [x.to_dict() for x in self.month_periods],
            'three_month': [x.to_dict() for x in self.three_month_periods],
            'six_month': [x.to_dict() for x in self.six_month_periods],
            'twelve_month': [x.to_dict() for x in self.year_periods],
            'created': self.created
        }

    def periods_in_range(self, months_per, start, end):
        one_day = datetime.timedelta(days=1)
        ends = []
        cur_start = start.replace(day=1)

        # in edge cases (all sids filtered out, start/end are adjacent)
        # a test will not generate any returns data
        if len(self.algorithm_returns) == 0:
            return ends

        #ensure that we have an end at the end of a calendar month, in case
        #the return series ends mid-month...
        the_end = end.replace(day=1) + relativedelta(months=1) - one_day
        while True:
            cur_end = cur_start + relativedelta(months=months_per) - one_day
            if(cur_end > the_end):
                break
            cur_period_metrics = RiskMetricsPeriod(
                start_date=cur_start,
                end_date=cur_end,
                returns=self.algorithm_returns,
                benchmark_returns=self.benchmark_returns
            )

            ends.append(cur_period_metrics)
            cur_start = cur_start + relativedelta(months=1)

        return ends
