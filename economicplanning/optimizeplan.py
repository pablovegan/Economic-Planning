"""
Create a class that optimizes a economy using linear programming.

Classes
-------
InfeasibleProblem
OptimizePlan
"""
from math import ceil
from typing import Optional

from cvxpy import Minimize, Problem, Variable
from numpy import array, ndarray, zeros


class InfeasibleProblem(Exception):
    """Exception raised for infeasible LP problems in the input salary.

    Args:
        iter (int): Current iteration of the linear programming algorithm.
    """

    def __init__(self, iter: int) -> None:
        self.message = (
            f"LP problem in iteration period {iter} couldn't"
            " be solved. You may try increasing the horizon periods"
            " or the initial excess production."
        )
        super().__init__(self.message)


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
        econ (dict[str, list[ndarray]]): The economy, which contains supply-use tables,
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
        econ (EconomicPlan): The economy, which contains supply-use tables, import prices...
        num_products (int): The number of products in the economy.
        num_units (int): The number of production units in the economy.
        worked_hours (list[ndarray]): Total worked hours in each period.
        planned_activity (list[ndarray]): The planned activity for the production units
            in each period.
        prod_planned (list[ndarray]): The planned production for each product in each period.
        prod_excess (list[ndarray]): The excess production at the end of each period.
        export_deficit (list[ndarray]): The export deficit at the end of each period.
        activity (list[Variable]): The activity variables of our LP problem, which correspond to the
            level of activation of each production unit.
        final_import (list[Variable]): The final imported products (variables) of our LP problem,
            which correspond to the level of activation of each production unit.
    """

    __slots__ = (
        "plan_periods",
        "horizon_periods",
        "revise_periods",
        "econ",
        "num_products",
        "num_units",
        "worked_hours",
        "planned_activity",
        "prod_excess",
        "prod_planned",
        "export_deficit",
        "activity",
        "iter_period",
        "constraints_dict",
        "__dict__",
    )

    def __init__(
        self,
        plan_periods: int,
        horizon_periods: int,
        revise_periods: int,
        econ: dict[str, list[ndarray]],
        constraints_dict: dict[str, bool] = {"export_constraints": True},
    ) -> None:
        self.plan_periods = plan_periods
        self.horizon_periods = horizon_periods
        self.revise_periods = revise_periods

        self.econ = econ
        self.num_products = econ["supply"][0].shape[0]
        self.num_units = econ["supply"][0].shape[1]

        self._assert_plan()  # Assert the time periods are compatible

        self.activity = self._activity
        self.final_import = self._final_import
        self.constraints_dict = constraints_dict

    def _assert_plan(self) -> None:
        "Assert that the time periods are compatible."
        if self.revise_periods > self.horizon_periods:
            raise ErrorRevisePeriods
        min_iter = ceil(self.plan_periods / self.revise_periods)
        min_periods = self.revise_periods * (min_iter - 1) + self.horizon_periods
        if min_periods > len(self.econ["supply"]):
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
            activity.append(Variable(self.num_units, name=f"activity_{i}"))
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
            final_import.append(Variable(self.num_products, name=f"final_import_{i}"))
        return final_import

    def __call__(
        self, prod_excess: Optional[ndarray] = None, export_deficit: Optional[float] = None
    ) -> None:
        """Optimize the plan over the specified periods and horizon.

        Args:
            prod_excess (ndarray, optional): The remaining production at the initial
                time period. Defaults to None.
            export_deficit (float, optional): The export deficit at the initial
                time period. Defaults to None.
        """
        self.worked_hours: list[ndarray] = []
        self.planned_activity: list[ndarray] = []
        self.prod_planned: list[ndarray] = []
        self.export_deficit: list[ndarray] = []
        self.prod_excess: list[ndarray] = []

        prod_excess = zeros(self.num_products) if prod_excess is None else prod_excess
        export_deficit = 0 if export_deficit is None else export_deficit

        for i in range(0, self.plan_periods, self.revise_periods):
            self.iter_period = i
            # Solve the linear programming problem
            self.optimize_period(prod_excess, export_deficit)
            # Excess production and export deficit initialization for the next iteration
            prod_excess = self.prod_excess[-1]
            if self.constraints_dict["export_constraints"] is True:
                export_deficit = self.export_deficit[-1]

        # Convert the plan solutions to numpy arrays
        self.worked_hours = array(self.worked_hours)
        self.planned_activity = array(self.planned_activity).T
        self.prod_planned = array(self.prod_planned).T
        self.prod_excess = array(self.prod_excess).T
        if self.constraints_dict["export_constraints"] is True:
            self.export_deficit = array(self.export_deficit).T

    def optimize_period(self, prod_excess: ndarray, export_deficit: float) -> None:
        """Optimize one period of the plan.

        Args:
            prod_excess (ndarray): The excess production at the end of each period.
            export_deficit (float): The export deficit at the end of each period.

        Raises:
            InfeasibleProblem: Exception raised for infeasible LP problems in the input salary.
        """
        problem = Problem(Minimize(self.cost), self.constraints(prod_excess, export_deficit))
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
            self.worked_hours[t] = self.worked_hours[t].value
            self.prod_excess[t] = self.prod_excess[t].value
            self.prod_planned[t] = self.prod_planned[t].value

            if self.constraints_dict["export_constraints"] is True:
                self.export_deficit[t] = self.export_deficit[t].value

    @property
    def cost(self) -> Variable:
        r"""Create the cost function to optimize and save the total worked hours in each period.
        $$    \text{minimize}\: \sum_{t=0}^T c_t \cdot x_t  $$

        Returns:
            Variable: Cost function to optimize.
        """
        # cost = Variable(1)
        # cost.value = 0
        cost = 0
        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            worked_hours = self.econ["worked_hours"][t] @ self.activity[t]
            import_prices = self.econ["prices_import"][t] @ self.final_import[t]
            # TODO: revise cost function. The more we penalize imports, the more hours we work
            cost += worked_hours + 2 * import_prices  # * Note: pennalize imports over domestic
            # Record the worked hours in each period
            if t <= self.iter_period + self.revise_periods - 1:
                self.worked_hours.append(worked_hours)
        return cost

    def constraints(self, prod_excess: ndarray, export_deficit: float) -> list:
        """Create a list of constraints for the plan and save
        the excess and planned production.

        Args:
            prod_excess (ndarray): The excess production at the end of each period.
            export_deficit (float): The export deficit at the end of each period.

        Returns:
            list: Constraints that define the optimization region.
        """
        constraints = self.positivity_constraints()
        constraints += self.production_constraints(prod_excess)
        if self.constraints_dict["export_constraints"] is True:
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

    def production_constraints(self, prod_excess: ndarray) -> list:
        r"""We must produce more than the target output,
        $$e_{t-1} + S_t \cdot x_t - U^\text{dom}_t \cdot x_t +
        f^\text{imp}_t \geq f^\text{exp}_t + f^\text{dom}_t \:.$$

        Args:
            prod_excess (ndarray): The excess production at the end of each period.

        Returns:
            list: Production meets target constraints.
        """
        constraints = []
        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            supply_use = self.econ["supply"][t] - self.econ["use_domestic"][t]
            prod_planned = supply_use @ self.activity[t]
            prod_excess = (
                self.econ["depreciation"] @ prod_excess
                + prod_planned
                + self.final_import[t]
                - self.econ["final_domestic"][t]
                - self.econ["final_export"][t]
            )
            constraints.append(prod_excess >= 0)
            # We record the planned prod, excess prod and trade deficit in the revised periods
            if t <= self.iter_period + self.revise_periods - 1 and t <= self.plan_periods - 1:
                self.prod_excess.append(prod_excess)
                self.prod_planned.append(prod_planned)
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
        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            total_export = self.econ["prices_export"][t] @ self.econ["final_export"][t]
            total_import = self.econ["prices_import"][t] @ (
                self.econ["use_import"][t] @ self.activity[t] + self.final_import[t]
            )
            export_deficit = export_deficit + total_export - total_import
            # constraints.append(export_deficit >= -1e6)  # We limit export deficit

            # We record the planned prod, excess prod and trade deficit in the revised periods
            if t <= self.iter_period + self.revise_periods - 1 and t <= self.plan_periods - 1:
                self.export_deficit.append(export_deficit)
        constraints.append(export_deficit >= 0)
        return constraints
