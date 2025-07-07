"""Optimize an Economy dataclass using linear programming.

Classes:
    InfeasibleProblem
    ErrorRevisePeriods
    ErrorPeriods
    OptimizePlan

TODO:
    Input targets as an object in optimizer?
        - Target domestic
        - Target export
    Add more ecological constraints

"""

import logging
from math import ceil

from cvxpy import Minimize, Maximize, Problem, Variable, Constraint, multiply
import numpy as np
from numpy.typing import NDArray

from .economy import Economy, TargetEconomy, PlannedEconomy
from .ecology import Ecology, TargetEcology, PlannedEcology


class InfeasibleProblem(Exception):
    """Exception raised for infeasible LP problems in the input salary.

    Args:
        iter_ (int): Current iteration of the linear programming algorithm.
    """

    def __init__(self, iter_: int) -> None:
        message = (
            f"LP problem in iteration period {iter_} couldn't"
            " be solved. You may try increasing the horizon periods"
            " or the initial surplus production."
        )
        super().__init__(message)


class ErrorRevisePeriods(Exception):
    """Error in the number of periods given."""

    def __init__(self) -> None:
        self.message = (
            "Number of revise periods must be less or equal the number of horizon periods."
        )
        super().__init__(self.message)


class ErrorPeriods(Exception):
    """Error in the number of revise periods given."""

    def __init__(self) -> None:
        self.message = (
            "The number of periods provided in the economy must be greater"
            "or equal to the number of periods we need to optimize our plan."
        )
        super().__init__(self.message)


def harmony(x):
    return -1 / (x)


class OptimizePlan:
    r"""Given the data for an economy, create the desired constraints and calculate
    the desired production for the upcoming years using linear programming and
    receding horizon control.

    To plan an economy, we need to solve the following linear programming problem

    $$    \text{minimize}\: \sum_{t=0}^T c_t \cdot x_t  $$

    subject to different constraints:

    1. The activity of the production units must be positive at each period,
        $$x_t \geq 0 \:.$$

    2. More is produced than it is consumed,
        $$e_{t-1} + S_t \cdot x_t - U^\text{dom}_t \cdot x_t +
        f^\text{imp}_t \geq f^\text{exp}_t + f^\text{dom}_t \:,$$

    3. Trade balance is positive after a certain number of periods,
        $$\sum_{t=1}^T\: (U^\text{imp}_t \cdot x_t + f^\text{imp}_t) \cdot p^\text{imp}
        \: \leq \: \sum_{t=1}^T \: f^\text{exp}_t \cdot p^\text{exp} \:. $$

    Args:
        periods (int): The number of periods to actually plan (discarding the horizon).
        horizon_periods (int): The number of periods to plan in each iteration.
        revise_periods (int): The number of periods after which to revise a plan.
        economy (EconomicPlan): The economy, which contains supply-use tables, import prices...

    Attributes:
        periods (int): The number of periods to actually plan (discarding the horizon).
            For example, we may want to plan the production for the next 4 years.
        horizon_periods (int): The number of periods to plan in each iteration.
            For example, we may want to use a horizon of 6 years.
        revise_periods (int): The number of periods after which to revise a plan.
            For example, if we planned a horizon of 6 years and we choose to revise
            the plan after 2 years, we discard the resting 4 years and plan again.
        economy (EconomicPlan): The economy, which contains supply-use tables, import prices...
        worked_hours (list[NDArray]): Total worked hours in each period.
        planned_activity (list[NDArray]): The planned activity for the production units
            in each period.
        planned_production (list[NDArray]): The planned production for each product in each period.
        planned_surplus (list[NDArray]): The surplus production at the end of each period.
        export_deficit (list[NDArray]): The export deficit at the end of each period.
        activity (list[Variable]): The activity variables of our LP problem, which correspond to the
            level of activation of each production unit.
        total_import (list[Variable]): The final imported products (variables) of our LP problem,
            which correspond to the level of activation of each production unit.
    """

    def __init__(
        self,
        economy: Economy,
        ecology: Ecology | None = None,
    ) -> None:
        self.economy = economy
        self.ecology = ecology
        self._validate_plan(economy)

        self.planned_economy = PlannedEconomy()
        self.planned_ecology = PlannedEcology()

    def _validate_plan(self, economy: Economy) -> None:
        "Validate that the time periods are compatible."
        if self.revise_periods > self.horizon_periods:
            logging.error("ErrorRevisePeriods exception raised.")
            raise ErrorRevisePeriods
        min_iter = ceil(self.periods / self.revise_periods)
        min_periods = self.revise_periods * (min_iter - 1) + self.horizon_periods
        if min_periods > len(economy.supply):
            logging.error("ErrorPeriods exception raised.")
            raise ErrorPeriods

    def _initialize_plan_variables(self):
        """Initialize variables needed for the optimization algorithm."""
        self.production = []
        self.surplus = []
        self.export_deficit = []
        self.worked_hours = []
        self.produced_pollutants = []
        self.activity = [  # Industrial activity is set as nonnegative
            Variable(self.economy.sectors, name=f"activity_{t}", nonneg=True)
            for t in range(self.periods + self.horizon_periods - 1)
        ]
        self.total_import = [  # Imports are set as nonnegative
            Variable(self.economy.products, name=f"total_import_{t}", nonneg=True)
            for t in range(self.periods + self.horizon_periods - 1)
        ]

    def __call__(
        self,
        periods: int,
        horizon_periods: int,
        revise_periods: int,
        target_economy: TargetEconomy,
        target_ecology: TargetEcology | None = None,
        init_surplus: NDArray | None = None,
        init_export_deficit: float = 0,
    ) -> PlannedEconomy:
        """Optimize the plan over the specified periods and horizon.

        Args:
            init_surplus (NDArray, optional): The surplus production at the initial
                time period. Defaults to None.
            init_export_deficit (float, optional): The export deficit at the initial
                time period. Defaults to None.
        """
        self._validate_plan(self.economy)
        self.periods: int = periods
        self.horizon_periods: int = horizon_periods
        self.revise_periods: int = revise_periods
        self._initialize_plan_variables()
        init_surplus = np.zeros(self.economy.products) if init_surplus is None else init_surplus

        self.optimize_period(0, target_economy, target_ecology, init_surplus, init_export_deficit)
        for period in range(self.revise_periods, self.periods, self.revise_periods):
            self.optimize_period(
                period,
                target_economy,
                target_ecology,
                self.planned_economy.surplus[-1],
                self.planned_economy.export_deficit[-1],
            )
        return self.planned_economy, self.planned_ecology

    def optimize_period(
        self,
        period: int,
        target_economy: TargetEconomy,
        target_ecology: TargetEcology,
        surplus: NDArray,
        export_deficit: float,
    ) -> None:
        """Optimize one period of the plan.

        Args:
            period (int): current period of the optimization.
            surplus (NDArray): The surplus production at the end of each period.
            export_deficit (float): The export deficit at the end of each period.

        Raises:
            InfeasibleProblem: Exception raised for infeasible LP problems in the input salary.
        """
        constraints = self.export_constraints(period, target_economy, export_deficit)
        # constraints += self.production_constraints(period, target_economy, surplus)
        constraints += self.labor_realloc_constraint(period)
        if self.ecology is not None:
            constraints += self.pollutants_constraint(period, target_ecology)
        # objective = Minimize(self.cost(period))
        objective = Minimize(self.cost_harmony(period, target_economy, surplus))
        problem = Problem(objective, constraints)
        problem.solve(verbose=False)

        if problem.status in ["infeasible", "unbounded"]:
            logging.error(f"Problem value is {problem.value}.")
            raise InfeasibleProblem(period)

        self._save_plan_period(period)

    def _save_plan_period(self, period: int):
        """Save the value of the quantities we are interested in"""
        for extra_periods in range(min(self.revise_periods, self.periods - period)):
            t = period + extra_periods
            self.planned_economy.activity.append(self.activity[t].value)
            self.planned_economy.production.append(self.production[t].value)
            self.planned_economy.surplus.append(self.surplus[t].value)
            self.planned_economy.total_import.append(self.total_import[t].value)
            self.planned_economy.export_deficit.append(self.export_deficit[t].value)
            self.planned_economy.worked_hours.append(self.worked_hours[t].value)
            self.planned_ecology.pollutants.append(self.produced_pollutants[t].value)

    def cost_harmony(
        self, period: int, target_economy: TargetEconomy, surplus: NDArray
    ) -> Variable:
        r"""Create the cost function to optimize and save the total worked hours in each period.
        $$    \text{minimize}\: \sum_{t=0}^T c_t \cdot x_t  $$
        Args:
            period (int): current period of the optimization.
        Returns:
            Variable: Cost function to optimize.
        """
        cost = 0
        for t in range(period, period + self.horizon_periods):
            supply_use = (
                self.economy.supply[t] - self.economy.use_domestic[t] - self.economy.use_import[t]
            )
            production = supply_use @ self.activity[t]
            final_planned = (
                self.economy.depreciation[t] @ surplus + production + self.total_import[t]
            )
            final_target = (
                target_economy.domestic[t] + target_economy.exports[t] + target_economy.imports[t]
            )
            surplus = final_planned - final_target

            # cost += harmony(sum(surplus / final_target))
            cost += sum(harmony(surplus) @ harmony(surplus))
            # Record the planned production and the surplus production in the revised periods
            if t <= period + self.revise_periods - 1 and t <= self.periods - 1:
                self.surplus.append(surplus)
                self.production.append(production)
                self.worked_hours.append(self.economy.worked_hours[t] @ self.activity[t])

        return cost

    def cost_hours(self, period: int) -> Variable:
        r"""Create the cost function to optimize and save the total worked hours in each period.
        $$    \text{minimize}\: \sum_{t=0}^T c_t \cdot x_t  $$
        Args:
            period (int): current period of the optimization.
        Returns:
            Variable: Cost function to optimize.
        """
        cost = 0
        for t in range(period, period + self.horizon_periods):
            worked_hours = self.economy.worked_hours[t] @ self.activity[t]
            import_prices = self.economy.prices_import[t] @ self.total_import[t]
            # TODO: revise cost function. The more we penalize imports, the more hours we work
            cost += worked_hours + import_prices

            if t <= period + self.revise_periods - 1:  # Record the worked hours in each period
                self.worked_hours.append(worked_hours)
        return cost

    def production_constraints(
        self, period: int, target_economy: TargetEconomy, surplus: NDArray
    ) -> list[Constraint]:
        r"""We must produce more than the target output,
        $$e_{t-1} + S_t \cdot x_t - U^\text{dom}_t \cdot x_t +
        f^\text{imp}_t \geq f^\text{exp}_t + f^\text{dom}_t \:.$$

        Args:
            period (int): current period of the optimization.
            surplus (NDArray): the surplus production at the end of each period.

        Returns:
            list: Production meets target constraints.
        """
        constraints = []
        for t in range(period, period + self.horizon_periods):
            supply_use = (
                self.economy.supply[t] - self.economy.use_domestic[t] - self.economy.use_import[t]
            )
            production = supply_use @ self.activity[t]
            surplus = (
                self.economy.depreciation[t] @ surplus
                + production
                + self.total_import[t]
                - target_economy.domestic[t]
                - target_economy.exports[t]
                - target_economy.imports[t]
            )
            constraints.append(surplus >= 0)
            # Record the planned production and the surplus production in the revised periods
            if t <= period + self.revise_periods - 1 and t <= self.periods - 1:
                self.surplus.append(surplus)
                self.production.append(production)
        return constraints

    def export_constraints(
        self, period: int, target_economy: TargetEconomy, export_deficit: float
    ) -> list[Constraint]:
        r"""We must export more than we import at the end of the horizon.

        $$ \: \sum_{t=1}^T \: f^\text{exp}_t \cdot p^\text{exp} \: \geq
        \: \sum_{t=1}^T\: (U^\text{imp}_t \cdot x_t + f^\text{imp}_t) \cdot p^\text{imp}\:. $$

        Note:
            If we don't force a positive deficit at the end of the revise period, we will have
            an ever increasing export deficit.

        Args:
            period (int): current period of the optimization.
            export_deficit (float): The export deficit at the end of each period.

        Returns:
            list: Export constraints.
        """
        constraints = []
        for t in range(period, period + self.horizon_periods):
            total_price_export = self.economy.prices_export[t] @ target_economy.exports[t]
            total_price_import = self.economy.prices_import[t] @ self.total_import[t]
            # economy.use_import[t] @ self.activity[t] + self.target_import[t]
            export_deficit = export_deficit + total_price_export - total_price_import
            # Save the trade deficit in the revised periods
            if t <= period + self.revise_periods - 1 and t <= self.periods - 1:
                self.export_deficit.append(export_deficit)

        # constraints.append(export_deficit <= 1e6)  # Limit export deficit
        constraints.append(export_deficit >= 0)
        return constraints

    def labor_realloc_constraint(self, period: int) -> list[Constraint]:
        r"""This constraint limits the reallocation of labor from one period to the
        next. For example, one cannot turn all farmers into train manufacturers in one year.

        Args:
            period (int): current period of the optimization.

        Returns:
            list: Labor reallocation constraints.
        """
        realloc_coef = 0.1
        realloc_low_limit = np.array([1 - realloc_coef] * self.economy.sectors)
        realloc_upper_limit = np.array([1 + realloc_coef] * self.economy.sectors)
        constraints = []
        for t in range(period, period + self.horizon_periods):
            if t == 0:  # No restrictions in the first period
                continue
            constraints.append(
                self.activity[t] >= multiply(realloc_low_limit, self.activity[t - 1])
            )
            constraints.append(
                self.activity[t] <= multiply(realloc_upper_limit, self.activity[t - 1])
            )
        return constraints

    def pollutants_constraint(self, period: int, target_ecology: TargetEcology) -> list[Constraint]:
        r"""Maximum pollution allowed."""
        constraints = []
        for t in range(period, period + self.horizon_periods):
            produced_pollutants = self.ecology.pollutant_sector[t] @ self.activity[t]
            constraints.append(produced_pollutants <= target_ecology.pollutants[t])
            # Record the planned production and the surplus production in the revised periods
            if t <= period + self.revise_periods - 1 and t <= self.periods - 1:
                self.produced_pollutants.append(produced_pollutants)
        return constraints
