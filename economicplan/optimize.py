"""
Create a class that optimizes a economy using linear programming.

Classes
-------
InfeasibleProblem
ErrorRevisePeriods
ErrorPeriods
OptimizePlan
"""
from math import ceil

from cvxpy import Minimize, Problem, Variable
from numpy import array, ndarray, zeros
from numpy.typing import NDArray

from economicplan import Economy

class InfeasibleProblem(Exception):
    """Exception raised for infeasible LP problems in the input salary.

    Args:
        iter (int): Current iteration of the linear programming algorithm.
    """

    def __init__(self, iter: int) -> None:
        message = (
            f"LP problem in iteration period {iter} couldn't"
            " be solved. You may try increasing the horizon periods"
            " or the initial surplus production."
        )
        super().__init__(message)


class ErrorRevisePeriods(Exception):
    """Error in the number of periods given."""

    def __init__(self) -> None:
        self.message = (
            "Number of revise periods must be less or equal the" "number of horizon periods."
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
        economy (dict[str, list[ndarray]]): The economy, which contains supply-use tables,
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
        worked_hours (list[ndarray]): Total worked hours in each period.
        planned_activity (list[ndarray]): The planned activity for the production units
            in each period.
        planned_production (list[ndarray]): The planned production for each product in each period.
        planned_surplus (list[ndarray]): The surplus production at the end of each period.
        export_deficit (list[ndarray]): The export deficit at the end of each period.
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
        economy: Economy,
    ) -> None:
        self.plan_periods = plan_periods
        self.horizon_periods = horizon_periods
        self.revise_periods = revise_periods

        self.economy = economy

        self._validate_plan()  # Assert the time periods are compatible

        self.activity = self._activity
        self.final_import = self._final_import

    def _validate_plan(self) -> None:
        "Validate that the time periods are compatible."
        if self.revise_periods > self.horizon_periods:
            raise ErrorRevisePeriods
        min_iter = ceil(self.plan_periods / self.revise_periods)
        min_periods = self.revise_periods * (min_iter - 1) + self.horizon_periods
        if min_periods > len(self.economy.supply):
            raise ErrorPeriods

    @property
    def _activity(self) -> list[Variable]:
        """Returns the unknown level of production of each unit for each period of the
        horizon plan, which are the activity we want to solve for in our problem.

        Returns:
            list[Variable]: Activity of each production unit that we want to optimize.
        """
        activity = []
        for i in range(self.plan_periods + self.horizon_periods - 1):
            activity.append(Variable(self.economy.sectors, name=f"activity_{i}"))
        return activity

    @property
    def _final_import(self) -> list[Variable]:
        """Returns the unknown level of production of each unit for each period of the
        horizon plan, which are the activity we want to solve for in our problem.

        Returns:
            list[Variable]: Activity of each production unit that we want to optimize.
        """
        final_import = []
        for i in range(self.plan_periods + self.horizon_periods - 1):
            final_import.append(Variable(self.economy.products, name=f"final_import_{i}"))
        return final_import

    def __call__(
        self, surplus: NDArray | None = None, export_deficit: float | None= None
    ) -> None:
        """Optimize the plan over the specified periods and horizon.

        Args:
            surplus (ndarray, optional): The surplus production at the initial
                time period. Defaults to None.
            export_deficit (float, optional): The export deficit at the initial
                time period. Defaults to None.
        """
        self.planned_activity: list[ndarray] = []
        self.planned_production: list[ndarray] = []
        self.planned_surplus: list[ndarray] = []
        self.planned_final_import: list[ndarray] = []
        self.export_deficit: list[ndarray] = []
        self.worked_hours: list[ndarray] = []

        surplus = zeros(self.economy.products) if surplus is None else surplus
        export_deficit = 0 if export_deficit is None else export_deficit

        for i in range(0, self.plan_periods, self.revise_periods):
            self.iter_period = i
            # Solve the linear programming problem
            self.optimize_period(surplus, export_deficit)
            # Surplus production and export deficit initialization for the next iteration
            surplus = self.planned_surplus[-1]
            export_deficit = self.export_deficit[-1]

        # Convert the plan solutions to numpy arrays
        self.planned_activity = array(self.planned_activity).T
        self.planned_production = array(self.planned_production).T
        self.planned_surplus = array(self.planned_surplus).T
        self.planned_final_import = array(self.planned_final_import).T
        self.export_deficit = array(self.export_deficit).T
        self.worked_hours = array(self.worked_hours)

    def optimize_period(self, surplus: ndarray, export_deficit: float) -> None:
        """Optimize one period of the plan.

        Args:
            surplus (ndarray): The surplus production at the end of each period.
            export_deficit (float): The export deficit at the end of each period.

        Raises:
            InfeasibleProblem: Exception raised for infeasible LP problems in the input salary.
        """
        problem = Problem(Minimize(self.cost), self.constraints(surplus, export_deficit))
        problem.solve(verbose=False)

        if problem.status in ["infeasible", "unbounded"]:
            print(f"Problem value is {problem.value}.")
            raise InfeasibleProblem(self.iter_period)

        # Get the value of the quantities we are interested in
        for r in range(self.revise_periods):
            t = self.iter_period + r
            if t > self.plan_periods - 1:
                break
            self.planned_activity.append(self.activity[t].value)
            self.planned_production[t] = self.planned_production[t].value
            self.planned_surplus[t] = self.planned_surplus[t].value
            self.planned_final_import.append(self.final_import[t].value)
            self.export_deficit[t] = self.export_deficit[t].value
            self.worked_hours[t] = self.worked_hours[t].value

    @property
    def cost(self) -> Variable:
        r"""Create the cost function to optimize and save the total worked hours in each period.
        $$    \text{minimize}\: \sum_{t=0}^T c_t \cdot x_t  $$

        Returns:
            Variable: Cost function to optimize.
        """
        cost = 0
        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            worked_hours = self.economy.worked_hours[t] @ self.activity[t]
            import_prices = self.economy.prices_import[t] @ self.final_import[t]
            # TODO: revise cost function. The more we penalize imports, the more hours we work
            cost += worked_hours + 2 * import_prices  # * Note: pennalize imports over domestic
            # Record the worked hours in each period
            if t <= self.iter_period + self.revise_periods - 1:
                self.worked_hours.append(worked_hours)
        return cost

    def constraints(self, surplus: ndarray, export_deficit: float) -> list:
        """Create a list of constraints for the plan and save
        the surplus and planned production.

        Args:
            surplus (ndarray): The surplus production at the end of each period.
            export_deficit (float): The export deficit at the end of each period.

        Returns:
            list: Constraints that define the optimization region.
        """
        constraints = self.positivity_constraints()
        constraints += self.production_constraints(surplus)
        constraints += self.export_constraints(export_deficit)
        return constraints

    def positivity_constraints(self) -> list:
        r"""Positivity constraints guarantee that production activity and final imported goods
        are positive,
        $$x_t \geq 0 \:.$$
        $$f^\text{imp}_t \geq 0 \:.$$

        Returns:
            list: Positive activity constraints.
        """
        constraints = []
        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            constraints.append(self.activity[t] >= 0)
            constraints.append(self.final_import[t] >= 0)
        return constraints

    def production_constraints(self, surplus: ndarray) -> list:
        r"""We must produce more than the target output,
        $$e_{t-1} + S_t \cdot x_t - U^\text{dom}_t \cdot x_t +
        f^\text{imp}_t \geq f^\text{exp}_t + f^\text{dom}_t \:.$$

        Args:
            surplus (ndarray): The surplus production at the end of each period.

        Returns:
            list: Production meets target constraints.
        """
        constraints = []
        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            supply_use = self.economy.supply[t] - self.economy.use_domestic[t]
            planned_production = supply_use @ self.activity[t]
            surplus = (
                self.economy.depreciation @ surplus
                + planned_production
                + self.final_import[t]
                - self.economy.final_domestic[t]
                - self.economy.final_export[t]
            )
            constraints.append(surplus >= 0)
            # We record the planned prod, surplus prod and trade deficit in the revised periods
            if t <= self.iter_period + self.revise_periods - 1 and t <= self.plan_periods - 1:
                self.planned_surplus.append(surplus)
                self.planned_production.append(planned_production)
        return constraints

    def export_constraints(self, export_deficit: float) -> list:
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
        # for t in range(self.iter_period, self.iter_period + self.horizon_periods):
        for t in range(self.iter_period, self.iter_period + self.revise_periods):
            total_price_export = self.economy.prices_export[t] @ self.economy.final_export[t]
            total_price_import = self.economy.prices_import[t] @ (
                self.economy.use_import[t] @ self.activity[t] + self.final_import[t]
            )
            export_deficit = export_deficit + total_price_export - total_price_import
            # constraints.append(export_deficit <= 1e6)  # We limit export deficit
            # constraints.append(export_deficit >= -1e6)  # We limit export deficit

            # We record the planned prod, surplus prod and trade deficit in the revised periods
            if t <= self.iter_period + self.revise_periods - 1 and t <= self.plan_periods - 1:
                self.export_deficit.append(export_deficit)

        constraints.append(export_deficit >= 0)
        return constraints
