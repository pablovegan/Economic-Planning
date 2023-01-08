"""Module docstring"""

from numpy import ndarray, zeros
from cvxpy import Variable, Problem, Minimize

from ..economy import Economy  # type: ignore


class OptimizePlan:
    """
    This subclass is responsible for the optimization of our economic plan.

    Attributes
    ----------
    plan_periods : int
        The number of periods to actually plan (discarding the horizon). For example,
        we may want to plan the production for the next 4 years.
    horizon_periods : int
        The number of periods to plan in each iteration. For example, we may want to use
        a horizon of 6 years.
    revise_periods : int
        The number of periods after which to revise a plan. For example, if we planned
        a horizon of 6 years and we choose to revise the plan after 2 years, we discard
        the resting 4 years and plan again. If we wanted to plan 4 years, we would need
        to do 2 horizon plans revised after 2 years.
    econ : EconomicPlan
        The economy, which contains supply-use tables, import tables...
    """

    __slots__ = ('horizon_periods', 'plan_periods', 'revise_periods',
                 'econ', 'num_units', 'variables', 'current_period',
                 'excess_prod', 'num_products', '__dict__')

    def __init__(self,
                 plan_periods: int,
                 horizon_periods: int,
                 revise_periods: int,
                 econ: Economy
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
        econ : EconomicPlan
            The economy, which contains supply-use tables, import tables...
        """
        self.plan_periods = plan_periods
        self.horizon_periods = horizon_periods
        self.revise_periods = revise_periods

        self.econ = econ
        self.num_products = econ.supply_use.shape[0]
        self.num_units = econ.supply_use.shape[1]

        self.variables = self._variables()

    def _variables(self) -> list[Variable]:
        """
        The unkwown variables in our model represent the level
        of production of each unit

        Returns
        -------
        list
            List of variables for each period of the horizon plan.
        """
        variables = []
        for i in range(self.plan_periods + self.horizon_periods - 1):
            variables.append(Variable(self.num_units, name=f'x{i}'))
        return variables

    def __call__(self, target_prod: list[ndarray]) -> tuple[list, list]:
        """
        Optimize the plan over the specified periods and horizon.

        Parameters
        ----------
        target_prod: list[ndarray]
            List of target productions for all planing periods.

        Returns
        -------
        tuple[list, list]
            The planned activity of each production unit and the total worked hours
            for each planned period.
        """
        planned_activity = []
        worked_hours = []
        self.excess_prod = zeros(self.num_products)

        # ? Cómo se relaciona un plan con el anterior???
        for t in range(0, self.plan_periods, self.revise_periods):
            self.current_period = t
            problem = self._optimize_period(target_prod)
            worked_hours.append(problem.value)
            self.excess_prod = self.excess_prod.value  # type: ignore

            for r in range(self.revise_periods):
                if self.variables[t + r].value is None:
                    raise ValueError('La variable es nula')
                planned_activity.append(self.variables[t + r].value)

        return planned_activity, worked_hours

    def _optimize_period(self, target_prod: list[ndarray]) -> Problem:
        """Optimize one period of the plan."""
        problem = Problem(Minimize(self._cost), self._constraints(target_prod))
        problem.solve(verbose=False)
        return problem

    def _constraints(self, target_prod: list[ndarray]) -> list:
        """Create a list of constraints for the plan."""
        constr_list = []
        total_import = 0
        total_export = 0

        excess_prod = self.excess_prod

        for i in range(self.current_period, self.current_period + self.horizon_periods):
            # Production must be positive
            constr_list.append(self.variables[i] >= 0)
            # We must produce more than the target
            excess_prod = (self.econ.deprec @ excess_prod +
                           self.econ.supply_use @ self.variables[i] - target_prod[i])
            constr_list.append(excess_prod >= 0)

            if i == self.current_period + self.revise_periods - 1:
                self.excess_prod = excess_prod

            # We must export more than we import
            total_import += self.econ.import_prices @ self.econ.use_imported @ self.variables[i]
            total_export += self.econ.export_prices @ self.econ.export_vector[i]
        constr_list.append(total_import <= total_export)

        return constr_list

    @property
    def _cost(self) -> Variable:
        """Create the cost function to optimize."""
        cost = 0
        for i in range(self.current_period, self.current_period + self.horizon_periods):
            cost += self.econ.cost_coefs @ self.variables[i]
        return cost  # type: ignore
