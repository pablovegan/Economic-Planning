"""
Create a class that optimizes a economy using linear programming.

Classes
-------
InfeasibleProblem
ErrorRevisePeriods
ErrorPeriods
OptimizePlan
"""
import logging
from math import ceil

from cvxpy import Minimize, Problem, Variable
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
        plan_periods (int): The number of periods to actually plan (discarding the horizon).
        horizon_periods (int): The number of periods to plan in each iteration.
        revise_periods (int): The number of periods after which to revise a plan.
        economy (dict[str, list[NDArray]]): The economy, which contains supply-use tables,
                import tables...
        constraints_dict (_type_, optional): _description_.
            Defaults to {"export_constraints": True}.

    Attributes:
        plan_periods (int): The number of periods to actually plan (discarding the horizon).
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
        plan_periods: int,
        horizon_periods: int,
        revise_periods: int,
    ) -> None:
        self.plan_periods: int = plan_periods
        self.horizon_periods: int = horizon_periods
        self.revise_periods: int = revise_periods
        self.activity_planned: list[NDArray] | None = None
        self.production_planned: list[NDArray] | None = None
        self.surplus_planned: list[NDArray] | None = None
        self.final_import_planned: list[NDArray] | None = None
        self.export_deficit: list[NDArray] | None = None
        self.worked_hours: list[NDArray] | None = None
        self.activity: list[Variable] | None = None
        self.final_import: list[Variable] | None = None

    def _validate_plan(self, economy: Economy) -> None:
        "Validate that the time periods are compatible."
        if self.revise_periods > self.horizon_periods:
            logging.error("ErrorRevisePeriods exception raised.")
            raise ErrorRevisePeriods
        min_iter = ceil(self.plan_periods / self.revise_periods)
        min_periods = self.revise_periods * (min_iter - 1) + self.horizon_periods
        if min_periods > len(economy.supply):
            logging.error("ErrorPeriods exception raised.")
            raise ErrorPeriods

    def __call__(
        self, economy: Economy, surplus: NDArray | None = None, export_deficit: float = 0
    ) -> PlannedEconomy:
        """Optimize the plan over the specified periods and horizon.

        Args:
            surplus (NDArray, optional): The surplus production at the initial
                time period. Defaults to None.
            export_deficit (float, optional): The export deficit at the initial
                time period. Defaults to None.
        """
        self._validate_plan(economy)  # Assert the time periods are compatible

        self.activity_planned = []
        self.production_planned = []
        self.surplus_planned = []
        self.final_import_planned = []
        self.export_deficit = []
        self.worked_hours = []
        self.activity = [
            Variable(economy.sectors, name=f"activity_{t}")
            for t in range(self.plan_periods + self.horizon_periods - 1)
        ]
        self.final_import = [
            Variable(economy.products, name=f"final_import_{t}")
            for t in range(self.plan_periods + self.horizon_periods - 1)
        ]
        surplus = np.zeros(economy.products) if surplus is None else surplus

        for period in range(0, self.plan_periods, self.revise_periods):
            self.optimize_period(period, economy, surplus, export_deficit)
            surplus = self.surplus_planned[-1]
            export_deficit = self.export_deficit[-1]

        return PlannedEconomy(
            activity_planned = np.array(self.activity_planned).T,
            production_planned = np.array(self.production_planned).T,
            surplus_planned = np.array(self.surplus_planned).T,
            final_import_planned = np.array(self.final_import_planned).T,
            export_deficit = np.array(self.export_deficit),
            worked_hours = np.array(self.worked_hours)
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
        problem = Problem(
            Minimize(self.cost(period, economy)),
            self.constraints(period, economy, surplus, export_deficit),
        )
        problem.solve(verbose=False)

        if problem.status in ["infeasible", "unbounded"]:
            logging.error(f"Problem value is {problem.value}.")
            raise InfeasibleProblem(period)

        # Get the value of the quantities we are interested in
        for extra_periods in range(min(self.revise_periods, self.plan_periods - period)):
            t = period + extra_periods
            # self.activity_planned.append(self.activity[t].value)
            self.activity_planned[t] = self.activity[t].value
            self.production_planned[t] = self.production_planned[t].value
            self.surplus_planned[t] = self.surplus_planned[t].value
            # self.final_import_planned.append(self.final_import[t].value)
            self.final_import_planned[t] = self.final_import[t].value
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
            cost += worked_hours + 2 * import_prices  # * Note: pennalize imports over domestic
            # Record the worked hours in each period
            if t <= period + self.revise_periods - 1:
                self.worked_hours.append(worked_hours)
        return cost

    def constraints(
        self, period: int, economy: Economy, surplus: NDArray, export_deficit: float
    ) -> list:
        """Create a list of constraints for the plan and save
        the surplus and planned production.

        Args:
            surplus (NDArray): The surplus production at the end of each period.
            export_deficit (float): The export deficit at the end of each period.

        Returns:
            list: Constraints that define the optimization region.
        """
        constraints = self.positivity_constraints(period)
        constraints += self.production_constraints(period, economy, surplus)
        constraints += self.export_constraints(period, economy, export_deficit)
        return constraints

    def positivity_constraints(self, period: int) -> list:
        r"""Positivity constraints guarantee that production activity and final imported goods
        are positive,
        $$x_t \geq 0 \:.$$
        $$f^\text{imp}_t \geq 0 \:.$$

        Returns:
            list: Positive activity constraints.
        """
        activity_constraints = [
            self.activity[t] >= 0 for t in range(period, period + self.horizon_periods)
        ]
        final_import_constraints = [
            self.final_import[t] >= 0 for t in range(period, period + self.horizon_periods)
        ]
        return activity_constraints + final_import_constraints

    def production_constraints(self, period: int, economy: Economy, surplus: NDArray) -> list:
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
            if t <= period + self.revise_periods - 1 and t <= self.plan_periods - 1:
                self.surplus_planned.append(surplus)
                self.production_planned.append(production_planned)
        return constraints

    def export_constraints(self, period: int, economy: Economy, export_deficit: float) -> list:
        r"""We must export more than we import at the end of the horizon.
        $$ \: \sum_{t=1}^T \: f^\text{exp}_t \cdot p^\text{exp} \: \geq
        \: \sum_{t=1}^T\: (U^\text{imp}_t \cdot x_t + f^\text{imp}_t) \cdot p^\text{imp}\:. $$

        Args:
            export_deficit (float): The export deficit at the end of each period.

        Returns:
            list: Export constraints.
        """
        constraints = []
        # * Force positive deficit at the end of the revise period
        # * or else we have an ever increasing export deficit.
        for t in range(period, period + self.revise_periods):
            total_price_export = economy.prices_export[t] @ economy.final_export[t]
            total_price_import = economy.prices_import[t] @ (
                economy.use_import[t] @ self.activity[t] + self.final_import[t]
            )
            export_deficit = export_deficit + total_price_export - total_price_import
            # constraints.append(export_deficit <= 1e6)  # Limit export deficit
            # constraints.append(export_deficit >= -1e6)  # Limit export deficit

            # Save the trade deficit in the revised periods
            if t <= period + self.revise_periods - 1 and t <= self.plan_periods - 1:
                self.export_deficit.append(export_deficit)

        constraints.append(export_deficit >= 0)
        return constraints
