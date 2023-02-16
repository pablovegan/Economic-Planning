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

    Attributes:
        message (str): Message to show when the exception is raised.
    """

    def __init__(self, iter: int) -> None:
        self.message = (
            f"LP problem in iteration period {iter} couldn't"
            " be solved. You may try increasing the horizon periods"
            " or the initial excess production."
        )
        super().__init__(self.message)


class OptimizePlan:
    r"""Given the data for an economy, create the desired constraints and calculate
    the desired production for the upcoming years using linear programming and
    receding horizon control.

    Adaptive Moment Estimation uses a step-dependent learning rate,
    a first moment :math:`a` and a second moment :math:`b`, reminiscent of
    the momentum and velocity of a particle:

    .. math::
        x^{(t+1)} = x^{(t)} - \eta^{(t+1)} \frac{a^{(t+1)}}{\sqrt{b^{(t+1)}} + \epsilon },

    where the update rules for the two moments are given by

    .. math::
        a^{(t+1)} &= \beta_1 a^{(t)} + (1-\beta_1) \nabla f(x^{(t)}),\\
        b^{(t+1)} &= \beta_2 b^{(t)} + (1-\beta_2) (\nabla f(x^{(t)}))^{\odot 2},\\
        \eta^{(t+1)} &= \eta \frac{\sqrt{(1-\beta_2^{t+1})}}{(1-\beta_1^{t+1})}.

    Above, :math:`( \nabla f(x^{(t-1)}))^{\odot 2}` denotes the element-wise square operation,
    which means that each element in the gradient is multiplied by itself. The hyperparameters
    :math:`\beta_1` and :math:`\beta_2` can also be step-dependent. Initially, the first and
    second moment are zero.

    The shift :math:`\epsilon` avoids division by zero.

    For more details, see `arXiv:1412.6980 <https://arxiv.org/abs/1412.6980>`_.


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
        variables (list[Variable]): The variables of our LP problem, which correspond to the
            level of activation of each production unit.
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
        "variables",
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
        self.num_products = econ["supply_use"][0].shape[0]
        self.num_units = econ["supply_use"][0].shape[1]

        self._assert_plan()  # Assert the time periods are compatible

        self.variables = self._variables
        self.constraints_dict = constraints_dict

    def _assert_plan(self) -> None:
        "Assert that the time periods are compatible."
        assert (
            self.revise_periods <= self.horizon_periods
        ), "Number of revise periods must be less or equal the number of horizon periods."

        min_iter = ceil(self.plan_periods / self.revise_periods)
        min_periods = self.revise_periods * (min_iter - 1) + self.horizon_periods
        assert min_periods <= len(self.econ["supply_use"]), (
            "The number of periods provided in the economy must be greater"
            "or equal to the number of periods we need to optimize our plan."
        )

    @property
    def _variables(self) -> list[Variable]:
        """Returns the unknown level of production of each unit for each period of the
        horizon plan, which are the variables we want to solve for in our problem.

        Returns
        -------
        list[Variable]
            Activity of each production unit that we want to optimize.
        """
        variables = []
        for i in range(self.plan_periods + self.horizon_periods - 1):
            variables.append(Variable(self.num_units, name=f"x{i}"))
        return variables

    def __call__(
        self, prod_excess: Optional[ndarray] = None, export_deficit: Optional[float] = None
    ) -> None:
        """
        Optimize the plan over the specified periods and horizon.

        Parameters
        ----------
        prod_excess : ndarray, optional
            The remaining production at the initial time period.
        export_deficit : ndarray, optional
            The export deficit at the initial time period.
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
            self._optimize_period(prod_excess, export_deficit)
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

    def _optimize_period(self, prod_excess: ndarray, export_deficit: float) -> None:
        """Optimize one period of the plan.

        Parameters
        ----------
        prod_excess : ndarray
            The excess production at the end of each period.
        export_deficit : float
            The export deficit at the end of each period.

        Raises
        ------
        InfeasibleProblem
            Exception raised for infeasible LP problems in the input salary.
        """
        problem = Problem(Minimize(self._cost), self._constraints(prod_excess, export_deficit))
        problem.solve(verbose=False)

        if problem.status in ["infeasible", "unbounded"]:
            print(f"Problem value is {problem.value}.")
            raise InfeasibleProblem(self.iter_period)

        # Get the value of the quantities we are interested in
        for r in range(self.revise_periods):
            t = self.iter_period + r
            if t > self.plan_periods - 1:
                break
            self.planned_activity.append(self.variables[t].value)
            self.worked_hours[t] = self.worked_hours[t].value
            self.prod_excess[t] = self.prod_excess[t].value
            self.prod_planned[t] = self.prod_planned[t].value

            if self.constraints_dict["export_constraints"] is True:
                self.export_deficit[t] = self.export_deficit[t].value

    @property
    def _cost(self) -> Variable:
        """Create the cost function to optimize and save
        the total worked hours in each period.

        Returns
        -------
        Variable
            Cost function to optimize.
        """
        cost = Variable(1)
        cost.value = 0
        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            worked_hours = self.econ["worked_hours"][t] @ self.variables[t]
            cost += worked_hours
            # Record the worked hours in each period
            if t <= self.iter_period + self.revise_periods - 1:
                self.worked_hours.append(worked_hours)
        return cost

    def _constraints(self, prod_excess: ndarray, export_deficit: float) -> list:
        """Create a list of constraints for the plan and save
        the excess and planned production.

        Parameters
        ----------
        prod_excess : ndarray
            The excess production at the end of each period.
        export_deficit : float
            The export deficit at the end of each period.

        Returns
        -------
        list
            Constraints that define the optimization region.
        """
        constraints = self._activity_constraints()
        constraints += self._production_constraints(prod_excess)
        if self.constraints_dict["export_constraints"] is True:
            constraints += self._export_constraints(export_deficit)
        return constraints

    def _activity_constraints(self) -> list:
        """Activity constraints guarantee that production activity is positive.

        Returns
        -------
        list
            Positive activity constraints.
        """
        constraints = []
        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            constraints.append(self.variables[t] >= 0)
        return constraints

    def _production_constraints(self, prod_excess: ndarray) -> list:
        """We must produce more than the target output.

        Parameters
        ----------
        prod_excess : ndarray
            The excess production at the end of each period.

        Returns
        -------
        list
            Production meets target constraints.
        """
        constraints = []
        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            supply_use = (
                self.econ["supply"][t] - self.econ["use_national"][t]
            )  # - self.econ["use_imported"][t]
            prod_planned = supply_use @ self.variables[t]
            prod_excess = (
                self.econ["depreciation"] @ prod_excess
                + prod_planned
                # + self.econ["imported_prod"][t]
                - self.econ["prod_target"][t]
            )
            constraints.append(prod_excess >= 0)
            # We record the planned prod, excess prod and trade deficit in the revised periods
            if t <= self.iter_period + self.revise_periods - 1 and t <= self.plan_periods - 1:
                self.prod_excess.append(prod_excess)
                self.prod_planned.append(prod_planned)
        return constraints

    def _export_constraints(self, export_deficit: float) -> list:
        """We must export more than we import at the end of the horizon.

        Parameters
        ----------
        export_deficit : float
            The export deficit at the end of each period.

        Returns
        -------
        list
            Export constraints.
        """
        # TODO: add a vector of imports for final use (not for production)
        constraints = []
        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            total_import = (
                self.econ["import_prices"][t] @ self.econ["use_imported"][t] @ self.variables[t]
            )
            total_export = self.econ["export_prices"][t] @ self.econ["prod_export"][t]
            export_deficit = export_deficit + total_export - total_import
            constraints.append(export_deficit >= -1e6)  # We limit export deficit

            # We record the planned prod, excess prod and trade deficit in the revised periods
            if t <= self.iter_period + self.revise_periods - 1 and t <= self.plan_periods - 1:
                self.export_deficit.append(export_deficit)
        constraints.append(export_deficit >= 0)
        return constraints
