"""
Create a class that optimizes a economy using linear programming.

Classes
-------
InfeasibleProblem
OptimizePlan
"""
from typing import Optional
from math import ceil

from numpy import ndarray, array, zeros
from cvxpy import Variable, Problem, Minimize


class InfeasibleProblem(Exception):
    """Exception raised for infeasible LP problems in the input salary."""

    def __init__(self, iter):
        self.message = (f"LP problem in iteration period {iter} couldn't"
                        " be solved. You may try increasing the horizon periods"
                        " or the initial excess production.")
        super().__init__(self.message)


class OptimizePlan:
    """
    Given the data for an economy, create the desired constraints and calculate
    the desired production for the upcoming years using linear programming and
    receding horizon control.

    Attributes
    ----------
    plan_periods : int
        The number of periods to actually plan (discarding the horizon).
        For example, we may want to plan the production for the next 4 years.
    horizon_periods : int
        The number of periods to plan in each iteration.
        For example, we may want to use a horizon of 6 years.
    revise_periods : int
        The number of periods after which to revise a plan.
        For example, if we planned a horizon of 6 years and we choose to revise
        the plan after 2 years, we discard the resting 4 years and plan again.
    econ : EconomicPlan
        The economy, which contains supply-use tables, import prices...
    num_products : int
        The number of products in the economy.
    num_units : int
        The number of production units in the economy.
    worked_hours : array_like
        Total worked hours in each period.
    planned_activity : array_like
        The planned activity for the production units in each period.
    planned_prod : array_like
        The planned production for each product in each period.
    excess_prod : array_like
        The excess production at the end of each period.
    export_deficit : array_like
        The export deficit at the end of each period.
    variables : list[Variable]
        The variables of our LP problem, which correspond to the
        level of activation of each production unit.
    """

    __slots__ = ('plan_periods', 'horizon_periods', 'revise_periods',
                 'econ', 'num_products', 'num_units', 'worked_hours',
                 'planned_activity', 'excess_prod', 'planned_prod',
                 'export_deficit', 'variables', 'iter_period', '__dict__')

    def __init__(self,
                 plan_periods: int,
                 horizon_periods: int,
                 revise_periods: int,
                 econ: dict
                 ):
        """
        Parameters
        ----------
        plan_periods : int
            The number of periods to actually plan (discarding the horizon).
        horizon_periods : int
            The number of periods to plan in each iteration.
        revise_periods : int
            The number of periods after which to revise a plan.
        econ : dict
            The economy, which contains supply-use tables, import tables...
        """
        self.plan_periods = plan_periods
        self.horizon_periods = horizon_periods
        self.revise_periods = revise_periods

        self.econ = econ
        self.num_products = econ['supply_use'][0].shape[0]
        self.num_units = econ['supply_use'][0].shape[1]

        self._assert_plan()  # Assert the time periods are compatible

        self.worked_hours = []
        self.planned_activity = []
        self.planned_prod = []
        self.export_deficit = []
        self.excess_prod = []

        self.variables = self._variables

    def _assert_plan(self):
        "Assert that the time periods are compatible."
        assert self.revise_periods <= self.horizon_periods, (
            "Number of revise periods must be less or equal the number of horizon periods.")

        min_iter = ceil(self.plan_periods / self.revise_periods)
        min_periods = self.revise_periods * (min_iter - 1) + self.horizon_periods
        assert min_periods <= len(self.econ['supply_use']), (
            "The number of periods provided in the economy must be greater"
            "or equal to the number of periods we need to optimize our plan.")

    @property
    def _variables(self) -> list[Variable]:
        """Returns the unknown level of production of each unit for each period of the
        horizon plan, which are the variables we want to solve for in our problem."""
        variables = []
        for i in range(self.plan_periods + self.horizon_periods - 1):
            variables.append(Variable(self.num_units, name=f'x{i}'))
        return variables

    def __call__(self,
                 excess_prod: Optional[ndarray] = None,
                 export_deficit: Optional[float] = None
                 ) -> None:
        """
        Optimize the plan over the specified periods and horizon.

        Parameters
        ----------
        excess_prod : arraylike
            The remaining production at the initial time period.
        export_deficit : arraylike
            The export deficit at the initial time period.

        Returns
        -------
        tuple[ndarray, ...]
            The planned activity of each production unit, the planned production
            and the total worked hours for each planned period.
        """
        excess_prod = zeros(self.num_products) if excess_prod is None else excess_prod
        export_deficit = zeros(self.num_products) if export_deficit is None else export_deficit

        for i in range(0, self.plan_periods, self.revise_periods):
            self.iter_period = i
            # Solve the linear programming problem
            self._optimize_period(excess_prod, export_deficit)
            # Get the value of the quantities we are interested in
            for r in range(self.revise_periods):
                t = i + r
                if t > self.plan_periods - 1:
                    break
                self.planned_activity.append(self.variables[t].value)
                self.worked_hours[t] = self.worked_hours[t].value
                self.excess_prod[t] = self.excess_prod[t].value
                self.planned_prod[t] = self.planned_prod[t].value
                self.export_deficit[t] = self.export_deficit[t].value

            # Excess production and export deficit initialization for the next iteration
            excess_prod = self.excess_prod[-1]
            export_deficit = self.export_deficit[-1]

        self.worked_hours = array(self.worked_hours)
        self.planned_activity = array(self.planned_activity).T
        self.planned_prod = array(self.planned_prod).T
        self.excess_prod = array(self.excess_prod).T
        self.export_deficit = array(self.export_deficit).T

    def _optimize_period(self, excess_prod: ndarray, export_deficit: float) -> None:
        """Optimize one period of the plan."""
        problem = Problem(Minimize(self._cost),
                          self._constraints(excess_prod, export_deficit))
        problem.solve(verbose=False)
        if problem.status in ["infeasible", "unbounded"]:
            print(f'Problem value is {problem.value}.')
            raise InfeasibleProblem(self.iter_period)

    @property
    def _cost(self) -> Variable:
        """Create the cost function to optimize and save
        the total worked hours in each period."""
        cost = 0
        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            worked_hours = self.econ['worked_hours'][t] @ self.variables[t]
            cost += worked_hours
            # Record the worked hours in each period
            if t <= self.iter_period + self.revise_periods - 1:
                self.worked_hours.append(worked_hours)
        return cost

    def _constraints(self, excess_prod: ndarray, export_deficit: float) -> list:
        """Create a list of constraints for the plan and save
            the excess and planned production."""
        constr_list = []

        for t in range(self.iter_period, self.iter_period + self.horizon_periods):
            # Production must be positive
            constr_list.append(self.variables[t] >= 0)
            # We must produce more than the target output
            planned_prod = self.econ['supply_use'][t] @ self.variables[t]
            excess_prod = (self.econ['depreciation'] @ excess_prod +
                           planned_prod - self.econ['target_output'][t])
            constr_list.append(excess_prod >= 0)
            # We must export more than we import at the end of the horizon
            total_import = (self.econ['import_prices'][t] @
                            self.econ['use_imported'][t] @ self.variables[t])
            total_export = self.econ['export_prices'][t] @ self.econ['export_output'][t]
            export_deficit = export_deficit + total_export - total_import
            # We limit export deficit
            constr_list.append(export_deficit >= -1e6)
            # We record the planned prod, excess prod and trade deficit in the revised periods
            if t <= self.iter_period + self.revise_periods - 1 and t <= self.plan_periods - 1:
                self.excess_prod.append(excess_prod)
                self.planned_prod.append(planned_prod)
                self.export_deficit.append(export_deficit)

        constr_list.append(export_deficit >= 0)
        return constr_list
