"""Optimize an Economy dataclass using linear programming.

Classes:
    InfeasibleProblem
    ErrorRevisePeriods
    ErrorPeriods
    OptimizePlan
"""

import logging
from math import ceil

from cvxpy import Minimize, Problem, Variable, Constraint
import numpy as np
from numpy.typing import NDArray

from economicplan.economy import Economy, PlannedEconomy


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
        economy (dict[str, list[NDArray]]): The economy, which contains supply-use tables,
                import tables...
        constraints_dict (_type_, optional): _description_.
            Defaults to {"export_constraints": True}.

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
        final_import (list[Variable]): The final imported products (variables) of our LP problem,
            which correspond to the level of activation of each production unit.
    """

    def __init__(
        self,
        periods: int,
        horizon_periods: int,
        revise_periods: int,
    ) -> None:
        self.periods: int = periods
        self.horizon_periods: int = horizon_periods
        self.revise_periods: int = revise_periods
        self.activity_planned: list[NDArray] = ...
        self.production_planned: list[NDArray] = ...
        self.surplus_planned: list[NDArray] = ...
        self.final_import_planned: list[NDArray] = ...
        self.export_deficit: list[NDArray] = ...
        self.worked_hours: list[NDArray] = ...
        self.activity: list[Variable] = ...
        self.final_import: list[Variable] = ...

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

    def __call__(
        self, economy: Economy, surplus: NDArray = ..., export_deficit: float = 0
    ) -> PlannedEconomy:
        """Optimize the plan over the specified periods and horizon.

        Args:
            surplus (NDArray, optional): The surplus production at the initial
                time period. Defaults to None.
            export_deficit (float, optional): The export deficit at the initial
                time period. Defaults to None.
        """
        self._validate_plan(economy)  # Assert the time periods are compatible
        # Initialize variables
        self.activity_planned = []
        self.production_planned = []
        self.surplus_planned = []
        self.final_import_planned = []
        self.export_deficit = []
        self.worked_hours = []
        self.activity = [  # Industrial activity is set as nonnegative
            Variable(economy.sectors, name=f"activity_{t}", nonneg=True)
            for t in range(self.periods + self.horizon_periods - 1)
        ]
        self.final_import = [  # Imports are set as nonnegative
            Variable(economy.products, name=f"final_import_{t}", nonneg=True)
            for t in range(self.periods + self.horizon_periods - 1)
        ]
        surplus = np.zeros(economy.products) if surplus is ... else surplus

        # Optimize the plan for each period
        for period in range(0, self.periods, self.revise_periods):
            self.optimize_period(period, economy, surplus, export_deficit)
            surplus = self.surplus_planned[-1]
            export_deficit = self.export_deficit[-1]

        return PlannedEconomy(
            activity=np.array(self.activity_planned).T,
            production=np.array(self.production_planned).T,
            surplus=np.array(self.surplus_planned).T,
            final_import=np.array(self.final_import_planned).T,
            export_deficit=np.array(self.export_deficit),
            worked_hours=np.array(self.worked_hours),
        )

    def optimize_period(
        self, period: int, economy: Economy, surplus: NDArray, export_deficit: float
    ) -> None:
        """Optimize one period of the plan.

        Args:
            surplus (NDArray): The surplus production at the end of each period.
            export_deficit (float): The export deficit at the end of each period.

        Raises:
            InfeasibleProblem: Exception raised for infeasible LP problems in the input salary.
        """
        constraints = self.production_constraints(period, economy, surplus)
        constraints += self.export_constraints(period, economy, export_deficit)

        objective = Minimize(self.cost(period, economy))
        problem = Problem(objective, constraints)
        problem.solve(verbose=False)

        if problem.status in ["infeasible", "unbounded"]:
            logging.error(f"Problem value is {problem.value}.")
            raise InfeasibleProblem(period)

        # Get the value of the quantities we are interested in
        for extra_periods in range(min(self.revise_periods, self.periods - period)):
            t = period + extra_periods
            self.activity_planned.append(self.activity[t].value)
            self.production_planned[t] = self.production_planned[t].value
            self.surplus_planned[t] = self.surplus_planned[t].value
            self.final_import_planned.append(self.final_import[t].value)
            self.export_deficit[t] = self.export_deficit[t].value
            self.worked_hours[t] = self.worked_hours[t].value

    def cost(self, period: int, economy: Economy) -> Variable:
        r"""Create the cost function to optimize and save the total worked hours in each period.
        $$    \text{minimize}\: \sum_{t=0}^T c_t \cdot x_t  $$

        Returns:
            Variable: Cost function to optimize.
        """
        cost = 0
        for t in range(period, period + self.horizon_periods):
            worked_hours = economy.worked_hours[t] @ self.activity[t]
            import_prices = economy.prices_import[t] @ self.final_import[t]
            # TODO: revise cost function. The more we penalize imports, the more hours we work
            cost += worked_hours + import_prices

            if t <= period + self.revise_periods - 1:  # Record the worked hours in each period
                self.worked_hours.append(worked_hours)
        return cost

    def production_constraints(
        self, period: int, economy: Economy, surplus: NDArray
    ) -> list[Constraint]:
        r"""We must produce more than the target output,
        $$e_{t-1} + S_t \cdot x_t - U^\text{dom}_t \cdot x_t +
        f^\text{imp}_t \geq f^\text{exp}_t + f^\text{dom}_t \:.$$

        Args:
            surplus (NDArray): The surplus production at the end of each period.

        Returns:
            list: Production meets target constraints.
        """
        constraints = []
        for t in range(period, period + self.horizon_periods):
            supply_use = economy.supply[t] - economy.use_domestic[t]
            production_planned = supply_use @ self.activity[t]
            surplus = (
                economy.depreciation[t] @ surplus
                + production_planned
                + self.final_import[t]
                - economy.final_domestic[t]
                - economy.final_export[t]
            )
            constraints.append(surplus >= 0)
            # Record the planned production and the surplus production in the revised periods
            if t <= period + self.revise_periods - 1 and t <= self.periods - 1:
                self.surplus_planned.append(surplus)
                self.production_planned.append(production_planned)
        return constraints

    def export_constraints(
        self, period: int, economy: Economy, export_deficit: float
    ) -> list[Constraint]:
        r"""We must export more than we import at the end of the horizon.

        $$ \: \sum_{t=1}^T \: f^\text{exp}_t \cdot p^\text{exp} \: \geq
        \: \sum_{t=1}^T\: (U^\text{imp}_t \cdot x_t + f^\text{imp}_t) \cdot p^\text{imp}\:. $$

        Note:
            If we don't force a positive deficit at the end of the revise period, we will have
            an ever increasing export deficit.

        Args:
            export_deficit (float): The export deficit at the end of each period.

        Returns:
            list: Export constraints.
        """
        constraints = []
        for t in range(period, period + self.revise_periods):
            total_price_export = economy.prices_export[t] @ economy.final_export[t]
            total_price_import = economy.prices_import[t] @ (
                economy.use_import[t] @ self.activity[t] + self.final_import[t]
            )
            export_deficit = export_deficit + total_price_export - total_price_import
            # Save the trade deficit in the revised periods
            if t <= period + self.revise_periods - 1 and t <= self.periods - 1:
                self.export_deficit.append(export_deficit)

        # constraints.append(export_deficit <= 1e6)  # Limit export deficit
        constraints.append(export_deficit >= 0)
        return constraints
