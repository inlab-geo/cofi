import numpy as np
import copy

from . import BaseInferenceTool, error_handler


class CoFIBorderCollieOptimization(BaseInferenceTool):
    r"""Implentation of a Border Collie Optimization Algorithm

    Based on the concepts and equations given in T. Dutta, S. Bhattacharyya, S. Dey and
    J. Platos, "Border Collie Optimization," in IEEE Access, vol. 8, pp. 109177-109197,
    2020, doi: 10.1109/ACCESS.2020.2999540 https://ieeexplore.ieee.org/document/9106341
    """

    documentation_links = []
    short_description = "CoFI's implemntation of Border Collie Optimization"

    @classmethod
    def required_in_problem(cls) -> set:
        return {"objective", "bounds", "model_shape"}

    @classmethod
    def optional_in_problem(cls) -> dict:
        return {"initial_model": []}

    @classmethod
    def required_in_options(cls) -> set:
        return set()

    @classmethod
    def optional_in_options(cls) -> dict:
        return {"number_of_iterations": 50, "flock_size": 50, "seed": 42}

    ### inner classes for dog and sheep

    class dog:
        def __init__(self, dim):
            self.pos = np.zeros(dim)
            self.vel = np.zeros(dim)
            self.acc = np.zeros(dim)
            self.tim = np.zeros(dim)
            self.fit = 0
            pass

    class sheep:
        def __init__(self, dim):
            self.pos = np.zeros(dim)
            self.vel = np.zeros(dim)
            self.acc = np.zeros(dim)
            self.tim = np.zeros(dim)
            self.eyed = 0
            self.fit = 0
            pass

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._components_used = list(self.required_in_problem())
        np.seterr(divide="ignore", invalid="ignore")
        # self model size
        self.mod_size = np.prod(inv_problem.model_shape)
        # lower and upper boundaries
        self.lower_bounds = []
        self.upper_bounds = []
        for i in range(self.mod_size):
            self.lower_bounds.append(inv_problem.bounds[i][0])
            self.upper_bounds.append(inv_problem.bounds[i][1])

        self.lower_bounds = np.array(self.lower_bounds)
        self.upper_bounds = np.array(self.upper_bounds)

        try:
            self.initial_model = inv_problem.initial_model
        except:
            self.initial_model = self.optional_in_problem()["initial_model"]
        self.rng = np.random.default_rng(self._params["seed"])
        # number of iterations
        self.number_of_iterations = self._params["number_of_iterations"]

    def __call__(self) -> dict:
        self.pack_size = 3
        self.pack = []
        self.flock_size = self._params["flock_size"]
        self.flock = []
        self.initialise()
        for itr in range(self.number_of_iterations):
            self.update_fitness()
            # print( self.itr_min_pos,"|",self.itr_min_fit)
            # determin if a sheep needs eyeing
            if itr > 0:
                for idx, sheep in enumerate(self.flock):
                    if self.flock_fit[idx] > self.flock_fit_prev[idx]:
                        sheep.eyed += 1

            if itr == 0 or fit_min > self.itr_min_fit:
                fit_min = self.itr_min_fit

                res = {
                    "success": "maybe",
                    "model": self.itr_min_pos,
                    "pack_fitness_history": self.pack_fit_hist,
                    "flock_fitness_history": self.flock_fit_hist,
                    "pack_position_history": self.pack_pos_hist,
                    "flock_position_history": self.flock_pos_hist,
                }

                dpos = []
                for dog in self.pack:
                    dpos.append(dog.pos)
                self.pack_pos_hist.append(dpos)
                spos = []

                for sheep in self.flock:
                    spos.append(sheep.pos)
                self.flock_pos_hist.append(spos)

                self.dog2sheep2dog()

                dpos = []
                for dog in self.pack:
                    dpos.append(dog.pos)
                self.pack_pos_hist.append(dpos)
                spos = []

                for sheep in self.flock:
                    spos.append(sheep.pos)
                self.flock_pos_hist.append(spos)

            else:
                self.dog2sheep2dog()

            self.update_movement()
            self.check_positions()

        return res

    # The following are useful functions used by the main optimiser but it makes sense
    # to keep them seperate

    def initialise(self):
        for i in range(self.pack_size):
            self.pack.append(self.dog(self.mod_size))
            if isinstance(self.initial_model, np.ndarray) or self.initial_model:
                self.pack[i].pos = self.initial_model
            else:
                self.pack[i].pos = (
                    self.rng.uniform(size=self.mod_size)
                    * (self.upper_bounds - self.lower_bounds)
                    + self.lower_bounds
                )
            self.pack[i].vel = self.rng.uniform(size=self.mod_size)
            self.pack[i].acc = self.rng.uniform(size=self.mod_size)
            self.pack[i].tim = self.rng.uniform(size=self.mod_size)
        for i in range(self.flock_size):
            self.flock.append(self.sheep(self.mod_size))
            if isinstance(self.initial_model, np.ndarray) or self.initial_model:
                self.flock[i].pos = self.initial_model
            else:
                self.flock[i].pos = (
                    np.random.rand(self.mod_size)
                    * (self.upper_bounds - self.lower_bounds)
                    + self.lower_bounds
                )
            self.flock[i].vel = self.rng.uniform(size=self.mod_size)
            self.flock[i].acc = self.rng.uniform(size=self.mod_size)
            self.flock[i].tim = self.rng.uniform(size=self.mod_size)
            # print(self.flock[i].pos)

        self.pack_fit = np.zeros(self.pack_size)
        self.flock_fit = np.zeros(self.flock_size)
        self.pop_fit = np.zeros(self.pack_size + self.flock_size)

        self.pack_fit_hist = []
        self.flock_fit_hist = []

        self.pack_pos_hist = []
        self.flock_pos_hist = []

        return

    def update_fitness(self):
        # could be done parallel
        self.flock_fit_prev = copy.copy(self.flock_fit)

        for i, dog in enumerate(self.pack):
            self.pack_fit[i] = self.inv_problem.objective(self.pack[i].pos)
            # print(i,self.pack_fit[i],self.pack[i].pos)
            self.pack[i].fit = self.pack_fit[i]
            self.pop_fit[i] = self.pack_fit[i]
        for i, sheep in enumerate(self.flock):
            self.flock_fit[i] = self.inv_problem.objective(self.flock[i].pos)
            self.flock[i].fit = self.flock_fit[i]
            self.pop_fit[i + self.pack_size] = self.flock_fit[i]

        self.pack_fit_min_idx = np.argmin(self.pack_fit)
        self.pack_fit_min_val = self.pack_fit[self.pack_fit_min_idx]

        self.flock_fit_min_idx = np.argmin(self.flock_fit)
        self.flock_fit_min_val = self.flock_fit[self.flock_fit_min_idx]

        self.pack_fit_hist.append(self.pack_fit)

        if self.flock_fit_min_val < self.pack_fit_min_val:
            self.itr_min_pos = self.flock[self.flock_fit_min_idx].pos
            self.itr_min_fit = self.flock_fit_min_val
        else:
            self.itr_min_pos = self.pack[self.pack_fit_min_idx].pos
            self.itr_min_fit = self.pack_fit_min_val

        return

    def dog2sheep2dog(self):
        idx = self.pop_fit.argsort()
        popl = copy.copy(self.pack + self.flock)

        for i, dog in enumerate(self.pack):
            self.pack[i].pos = copy.copy(popl[idx[i]].pos)
            self.pack[i].vel = copy.copy(popl[idx[i]].vel)
            self.pack[i].acc = copy.copy(popl[idx[i]].acc)
            self.pack[i].tim = copy.copy(popl[idx[i]].tim)
            self.pack[i].fit = copy.copy(popl[idx[i]].fit)

        for i, sheep in enumerate(self.flock):
            self.flock[i].pos = copy.copy(popl[idx[i + self.pack_size]].pos)
            self.flock[i].vel = copy.copy(popl[idx[i + self.pack_size]].vel)
            self.flock[i].acc = copy.copy(popl[idx[i + self.pack_size]].acc)
            self.flock[i].tim = copy.copy(popl[idx[i + self.pack_size]].tim)
            self.flock[i].fit = copy.copy(popl[idx[i + self.pack_size]].fit)
        return

    def update_movement(self):
        # tournament selection for choosing left and right dogs
        if np.random.randint(1, high=3) == 1:
            dog = copy.copy(self.pack[1])
            self.pack[2] = copy.copy(self.pack[1])
            self.pack[1] = copy.copy(dog)

        # update dog velocity, acceleration and time
        for i, dog in enumerate(self.pack):
            v0 = self.pack[i].vel
            ## self.pack[i].vel=self.pack[i].vel+self.pack[i].acc
            try:
                self.pack[i].vel = np.sqrt(
                    self.pack[i].vel ** 2 + 2.0 * self.pack[i].acc * self.pack[i].pos
                )
                assert not np.any(np.isnan(self.pack[i].vel))
            except:
                self.pack[i].vel = np.zeros(np.shape(self.pack[i].pos))
            self.pack[i].acc = (self.flock[i].vel - v0) / self.pack[i].tim
            self.pack[i].tim = np.mean((self.pack[i].vel - v0) / self.pack[i].tim)

        # update dog position
        for i, dog in enumerate(self.pack):
            ## self.pack[i].pos=self.pack[i].pos+self.pack[i].vel
            self.pack[i].pos = (
                self.pack[i].vel * self.pack[i].tim
                + 0.5 * self.pack[i].acc * self.pack[i].tim ** 2
            )

        # update sheep velocity acceleration and time
        for i, sheep in enumerate(self.flock):
            dg = (self.pack[0].fit - self.flock[i].fit) - (
                ((self.pack[1].fit + self.pack[2].fit) / 2.0) - self.flock[i].fit
            )
            v0 = self.flock[i].vel
            if dg >= 0.0:  # sheep is gathered
                ## self.flock[i].vel=self.pack[0].vel+self.pack[0].acc/self.pack[0].tim
                try:
                    self.flock[i].vel = np.sqrt(
                        self.flock[i].vel + 2.0 * self.flock[i].acc * self.flock[i].pos
                    )
                    assert not np.any(np.isnan(self.flock[i].vel))
                except:
                    self.flock[i].vel = np.zeros(np.shape(self.flock[i].pos))

            if dg < 0.0:
                if self.flock[i].eyed < 5:
                    tan_theta = np.tan(self.rng.integers(1, high=90))
                    ##vl = self.pack[1].vel*tan_theta+self.pack[1].acc
                    vl = np.sqrt(
                        (self.pack[1].vel * tan_theta) ** 2
                        + 2.0 * self.pack[1].acc * self.pack[1].pos
                    )
                    tan_theta = np.tan(self.rng.integers(91, high=180))
                    ##vr = self.pack[2].vel*tan_theta+self.pack[2].acc
                    vr = np.sqrt(
                        (self.pack[2].vel * tan_theta) ** 2
                        + 2.0 * self.pack[2].acc * self.pack[2].pos
                    )
                    vs = (vl + vr) / 2.0
                    self.flock[i].vel = vs

                else:
                    self.flock[i].eyed = 0
                    if self.pack[1].fit < self.pack[2].fit:
                        ## self.flock[i].vel=self.pack[2].vel-self.pack[2].acc/self.pack[2].tim
                        self.flock[i].vel = np.sqrt(
                            self.pack[2].vel ** 2
                            - 2.0 * self.pack[2].acc * self.pack[2].pos
                        )
                    else:
                        ## self.flock[i].vel=self.pack[1].vel-self.pack[1].acc/self.pack[1].tim
                        self.flock[i].vel = np.sqrt(
                            self.pack[1].vel ** 2
                            - 2.0 * self.pack[1].acc * self.pack[1].pos
                        )
            self.flock[i].acc = (self.flock[i].vel - v0) / self.flock[i].tim
            self.flock[i].tim = np.mean((self.flock[i].vel - v0) / self.flock[i].acc)

            if dg >= 0.0:
                self.flock[i].pos = (
                    self.flock[i].vel * self.flock[i].tim
                    + 0.5 * self.flock[i].acc * self.flock[i].tim ** 2
                )
            if dg < 0.0:
                self.flock[i].pos = (
                    self.flock[i].vel * self.flock[i].tim
                    - 0.5 * self.flock[i].acc * self.flock[i].tim ** 2
                )
        return

    def check_positions(self):
        for i in range(self.pack_size):
            for j in range(self.mod_size):
                if (
                    self.pack[i].pos[j] >= self.upper_bounds[j]
                    or self.pack[i].pos[j] <= self.lower_bounds[j]
                ):
                    self.pack[i].pos[j] = (
                        self.rng.uniform()
                        * (self.upper_bounds[j] - self.lower_bounds[j])
                        + self.lower_bounds[j]
                    )
                    self.pack[i].acc[j] = self.rng.uniform()
                    self.pack[i].tim = self.rng.uniform()
                if np.isnan(self.pack[i].acc[j]) or self.pack[i].acc[j] == 0:
                    self.pack[i].pos[j] = (
                        self.rng.uniform()
                        * (self.upper_bounds[j] - self.lower_bounds[j])
                        + self.lower_bounds[j]
                    )
                    self.pack[i].acc[j] = self.rng.uniform()
                    self.pack[i].tim = self.rng.uniform()
                if np.isnan(self.pack[i].vel[j]) or self.pack[i].vel[j] == 0:
                    self.pack[i].pos[j] = (
                        self.rng.uniform()
                        * (self.upper_bounds[j] - self.lower_bounds[j])
                        + self.lower_bounds[j]
                    )
                    self.pack[i].acc[j] = self.rng.uniform()
                    self.pack[i].tim = self.rng.uniform()
                if np.isnan(self.pack[i].tim) or self.pack[i].tim == 0:
                    self.pack[i].pos[j] = (
                        self.rng.uniform()
                        * (self.upper_bounds[j] - self.lower_bounds[j])
                        + self.lower_bounds[j]
                    )
                    self.pack[i].acc[j] = self.rng.uniform()
                    self.pack[i].tim = self.rng.uniform()

        for i in range(self.flock_size):
            for j in range(self.mod_size):
                if (
                    self.flock[i].pos[j] >= self.upper_bounds[j]
                    or self.flock[i].pos[j] <= self.lower_bounds[j]
                ):
                    self.flock[i].pos[j] = (
                        self.rng.uniform()
                        * (self.upper_bounds[j] - self.lower_bounds[j])
                        + self.lower_bounds[j]
                    )
                    self.flock[i].acc[j] = self.rng.uniform()
                    self.flock[i].tim = self.rng.uniform()
                if np.isnan(self.flock[i].acc[j]) or self.flock[i].acc[j] == 0:
                    self.flock[i].pos[j] = (
                        self.rng.uniform()
                        * (self.upper_bounds[j] - self.lower_bounds[j])
                        + self.lower_bounds[j]
                    )
                    self.flock[i].acc[j] = self.rng.uniform()
                    self.flock[i].tim = self.rng.uniform()
                if np.isnan(self.flock[i].vel[j]) or self.flock[i].vel[j] == 0:
                    self.flock[i].pos[j] = (
                        self.rng.uniform()
                        * (self.upper_bounds[j] - self.lower_bounds[j])
                        + self.lower_bounds[j]
                    )
                    self.flock[i].acc[j] = self.rng.uniform()
                    self.flock[i].tim = self.rng.uniform()
                if np.isnan(self.flock[i].tim) or self.flock[i].tim == 0:
                    self.flock[i].pos[j] = (
                        self.rng.uniform()
                        * (self.upper_bounds[j] - self.lower_bounds[j])
                        + self.lower_bounds[j]
                    )
                    self.flock[i].acc[j] = self.rng.uniform()
                    self.flock[i].tim = self.rng.uniform()
        return


# CoFI -> Ensemble methods -> Direct search -> Monte Carlo -> cofi.border_collie_optimization -> Border Collie Optimization Algorithm
# description: Implementation of a Border Collie Optimization Algorithm, based on the concepts and equations given in Dutta et al., IEEE Access, 2020.
# documentation: https://ieeexplore.ieee.org/document/9106341
