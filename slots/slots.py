"""
slots

A Python library to perform multi-armed bandit analyses.

Scenarios:
    - Run MAB test on simulated data (N bandits), default epsilon-greedy test.
        mab = slots.MAB(probs = [0.1,0.15,0.05])
        mab.run(trials = 10000)
        mab.best  # Bandit with highest probability after T trials

    - Run MAB test on "real" payout data (probabilites unknown).
        mab = slots.MAB(payouts = [0,0,0,1,0,0,0,0,0,....])
        mab.run(trials = 10000) # Max is length of payouts
"""


import numpy as np


class MAB(object):
    """
    Multi-armed bandit test class.
    """

    def __init__(
        self,
        num_bandits=3,
        probs=None,
        payouts=None,
        live=True,
        stop_criterion={"criterion": "regret", "value": 0.1},
    ):
        """
        Parameters
        ----------
        num_bandits : int
            default is 3
        probs : np.array of floats
            payout probabilities
        payouts : np.array of floats
            If `live` is True, `payouts` should be None.
        live : bool
            Whether the use is for a live, online trial.
        stop_criterion : dict
            Stopping criterion (str) and threshold value (float).
        """
        self.num_bandits = num_bandits
        self.choices = []
        self.payout_values = []
        if not probs:
            if not payouts:
                if live:
                    self.bandits = Bandits(
                        live=True, payouts=np.zeros(num_bandits), probs=None
                    )
                else:
                    self.bandits = Bandits(
                        probs=[np.random.rand() for x in range(num_bandits)],
                        payouts=np.ones(num_bandits),
                        live=False,
                    )
            else:

                self.bandits = Bandits(
                    probs=[np.random.rand() for x in range(len(payouts))],
                    payouts=payouts,
                    live=False,
                )
                num_bandits = len(payouts)
        else:
            if payouts:
                self.bandits = Bandits(probs=probs, payouts=payouts, live=False)
                num_bandits = len(payouts)
            else:
                self.bandits = Bandits(
                    probs=probs, payouts=np.ones(len(probs)), live=False
                )
                num_bandits = len(probs)

        self.wins = np.zeros(num_bandits)
        self.pulls = np.zeros(num_bandits)

        # Set the stopping criteria
        self.criteria = {"regret": self.regret_met}
        self.criterion = stop_criterion.get("criterion", "regret")
        self.stop_value = stop_criterion.get("value", 0.1)

        # Bandit selection strategies
        self.strategies = ["eps_greedy", "softmax", "ucb", "bayesian"]

    def run(self, trials=100, strategy=None, parameters=None):
        """
        Run MAB test with T trials.

        Parameters
        ----------
        trials : int
            Number of trials to run.
        strategy : str
            Name of selected strategy.
        parameters : dict
            Parameters for selected strategy.

        Available strategies:
            - Epsilon-greedy ("eps_greedy")
            - Softmax ("softmax")
            - Upper confidence bound ("ucb")

        Returns
        -------
        None
        """

        if trials < 1:
            raise Exception("MAB.run: Number of trials cannot be less than 1!")
        if not strategy:
            strategy = "eps_greedy"
        else:
            if strategy not in self.strategies:
                raise Exception(
                    "MAB,run: Strategy name invalid. Choose from:"
                    " {}".format(", ".join(self.strategies))
                )

        # Run strategy
        for n in range(trials):
            self._run(strategy, parameters)

    def _run(self, strategy, parameters=None):
        """
        Run single trial of MAB strategy.

        Parameters
        ----------
        strategy : function
        parameters : dict

        Returns
        -------
        None
        """

        choice = self.run_strategy(strategy, parameters)
        self.choices.append(choice)
        payout = self.bandits.pull(choice)
        self.payout_values.append(payout)
        if payout is None:
            print("Trials exhausted. No more values for bandit", choice)
            return None
        else:
            self.wins[choice] += payout
        self.pulls[choice] += 1

    def run_strategy(self, strategy, parameters):
        """
        Run the selected strategy and retrun bandit choice.

        Parameters
        ----------
        strategy : str
            Name of MAB strategy.
        parameters : dict
            Strategy function parameters

        Returns
        -------
        int
            Bandit arm choice index
        """

        return self.__getattribute__(strategy)(params=parameters)

    # ###### ----------- MAB strategies ---------------------------------------####
    def max_mean(self, params):
        """
        Pick the bandit with the current best observed proportion of winning.
        This is literally the same thing as the best() function below but for the sake of separating strategies...

        Returns
        -------
        int
            Index of chosen bandit
        """

        return np.argmax(est_payouts(params))

    def bayesian(self, params=None):
        """
        Run the Bayesian Bandit algorithm which utilizes a beta distribution
        for exploration and exploitation.

        Parameters
        ----------
        params : None
            For API consistency, this function can take a parameters argument,
            but it is ignored.

        Returns
        -------
        int
            Index of chosen bandit
        """
        # yeah I'll be totally honest, I'm not sure what's going on in this one, so this one isn't getting the non stationary stuff
        p_success_arms = [
            np.random.beta(self.wins[i] + 1, self.pulls[i] - self.wins[i] + 1)
            for i in range(len(self.wins))
        ]

        return np.array(p_success_arms).argmax()

    def eps_greedy(self, params):
        """
        Run the epsilon-greedy strategy and update self.max_mean()

        Parameters
        ----------
        Params : dict
            Epsilon
            step_size : None
                If a step_size is selected, we're using the sliding step size detailed in https://github.com/dquail/NonStationaryBandit
            sliding_window : None
                sliding_window just means that you're only using the last x results to do all future calculations. 
        Returns
        -------
        int
            Index of chosen bandit
        """
        # I'm sure it's theoretically possible to overlap sliding window and step size but that seems like real math so...
        if params and type(params) == dict:
            eps = params.get("epsilon")
        else:
            eps = 0.1

        r = np.random.rand()

        if r < eps:
            return np.random.choice(
                list(set(range(len(self.wins))) - {self.best(params)})
            )
        else:
            return self.best(params)

    def softmax(self, params):
        """
        Run the softmax selection strategy.

        Parameters
        ----------
        Params : dict
            Tau
            step_size : None
                If a step_size is selected, we're using the sliding step size detailed in https://github.com/dquail/NonStationaryBandit
            sliding_window : None
                sliding_window just means that you're only using the last x results to do all future calculations. 
        Returns
        -------
        int
            Index of chosen bandit
        """

        default_tau = 0.1

        if params and type(params) == dict:
            tau = params.get("tau")
            try:
                float(tau)
            except ValueError:
                "slots: softmax: Setting tau to default"
                tau = default_tau
        else:
            tau = default_tau

        # Handle cold start. Not all bandits tested yet.
        if True in (self.pulls < 3):
            return np.random.choice(range(len(self.pulls)))
        else:
            payouts = self.est_payouts(params)
            norm = sum(np.exp(payouts / tau))

        ps = np.exp(payouts / tau) / norm

        # Randomly choose index based on CMF
        cmf = [sum(ps[: i + 1]) for i in range(len(ps))]

        rand = np.random.rand()

        found = False
        found_i = None
        i = 0
        while not found:
            if rand < cmf[i]:
                found_i = i
                found = True
            else:
                i += 1

        return found_i

    def ucb(self, parameters):
        """
        Run the upper confidence bound MAB selection strategy.

        This is the UCB1 algorithm described in
        https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf

        Parameters
        ----------
        params : 
            step_size : float
                If a step_size is selected, we're using the sliding step size detailed in https://github.com/dquail/NonStationaryBandit
                NOTE THAT IN MY HUMBLE OPINION THEIR IMPLEMENTATION MAKES NO SENSE.
                THE NUMBER OF PULLS WHICH LEADS TO CONFIDENCE BOUND ISNT LINEAR ANYMORE IF YOU'RE NOT WEIGHING EACH PULL EVENLY RIGHT?
                Algorithm 2 in https://arxiv.org/pdf/0805.3415.pdf gives an equation for UCB with a step size but honestly that shit looks complicated.
                It also has the classic "where sigma is some appropriate constant" which is like... okay? 
            sliding_window: int
                Greater than 1, the number of most recent trials to use when calculating results
            
        Returns
        -------
        int
            Index of chosen bandit
        """

        # UCB = j_max(payout_j + sqrt(2ln(n_tot)/n_j))

        # Handle cold start. Not all bandits tested yet.

        if params and type(params) == dict:
            sliding_window = params.get("sliding_window")

        if sliding_window:
            choices = self.choices[-sliding_window:]
            payout_values = self.payout_values[-sliding_window:]
            wins = [0] * self.num_bandits
            pulls = [0] * self.num_bandits
            for x in sorted(set(choices)):
                indices = np.where(choices == x)[0]
                payouts = [payout_values[i] for i in indices]
                wins[x] = np.sum(payouts)
                pulls[x] = len(payouts)
        else:
            choices = self.choices
            payout_values = self.payout_values
            wins = self.wins
            pulls = self.pulls

        if True in (self.pulls < 3):
            return np.random.choice(range(len(self.pulls)))
        else:
            n_tot = sum(pulls)
            payouts = self.est_payouts(params)
            ubcs = payouts + np.sqrt(2 * np.log(n_tot) / pulls)

            return np.argmax(ubcs)

    def ucbtuned(self, parameters):
        """
        Run the upper confidence bound MAB selection strategy.

        This is the UCB1 algorithm described in
        https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf

        Parameters
        ----------
        params : None
            For API consistency, this function can take a parameters argument,
            but it is ignored.
        step_size : None
            If a step_size is selected, we're using the sliding step size detailed in https://github.com/dquail/NonStationaryBandit
            NOTE THAT IN MY HUMBLE OPINION THEIR IMPLEMENTATION MAKES NO SENSE.
            THE NUMBER OF PULLS WHICH LEADS TO CONFIDENCE BOUND ISNT LINEAR ANYMORE IF YOU'RE NOT WEIGHING EACH PULL EVENLY RIGHT?
            Algorithm 2 in https://arxiv.org/pdf/0805.3415.pdf gives an equation for UCB with a step size but honestly that shit looks complicated.
            It also has the classic "where sigma is some appropriate constant" which is like... okay? 
            
        Returns
        -------
        int
            Index of chosen bandit
        """
        # Variance_Adjusted = var(payouts) - payouts^2 + sqrt(2*ln(n_tot)/n_j)
        # UCBtuned = j_max(payout_j + sqrt(ln(n_tot)/n_j) * min(1/4, Variance_Adjusted)
        # note - 1/4 is the maximum of the variance adjsuted Bernoulli Random Variable Variance
        # further note - I don't think anyone has mathematically proven that UCB Tuned is better than UCB yet. It's been 16 years.

        # Handle cold start. Not all bandits tested yet.

        if params and type(params) == dict:
            sliding_window = params.get("sliding_window")

        if sliding_window:
            choices = self.choices[-sliding_window:]
            payout_values = self.payout_values[-sliding_window:]
            wins = [0] * self.num_bandits
            pulls = [0] * self.num_bandits
            for x in sorted(set(choices)):
                indices = np.where(choices == x)[0]
                payouts = [payout_values[i] for i in indices]
                wins[x] = np.sum(payouts)
                pulls[x] = len(payouts)
        else:
            choices = self.choices
            payout_values = self.payout_values
            wins = self.wins
            pulls = self.pulls

        if True in (self.pulls < 3):
            return np.random.choice(range(len(self.pulls)))
        else:
            n_tot = sum(pulls)
            payouts = self.est_payouts(params)
            variances = [0] * self.num_bandits
            exploration = np.log(n_tot) / pulls
            for x in sorted(set(choices)):
                indices = np.where(choices == x)[0]
                full_payouts = [payout_values[i] for i in indices]
                variance = np.var(full_payouts)
                variances[x] = variance
            variances_adjusted = np.subtract(variances, [i ** 2 for i in payouts])
            variances_adjusted = [
                i + np.sqrt(2 * exploration) for i in variances_adjusted
            ]
            ubctuneds = np.add(
                payouts,
                [(exploration * min(.25, i)) ** 0.5 for i in variances_adjusted],
            )

            return np.argmax(ubctuneds)

    # ###------------------------------------------------------------------####

    def best(self, parameters):
        """
        Return current 'best' choice of bandit.
        Parameters
        ----------
        step_size : None
            If a step_size is selected, we're using the sliding step size detailed in https://github.com/dquail/NonStationaryBandit
            
        Returns
        -------
        int
            Index of bandit
        """
        # is this not the same thing as max_mean...?
        if len(self.choices) < 1:
            print("slots: No trials run so far.")
            return None
        else:
            return np.argmax(est_payouts(parameters))

    def current(self):
        """
        Return last choice of bandit.

        Returns
        -------
        int
            Index of bandit
        """

        if len(self.choices) < 1:
            print("slots: No trials run so far.")
            return None
        else:
            return self.choices[-1]

    def est_payouts(self, bandit=None, params):
        """
        Calculate current estimate of average payout for each bandit.
        
        Parameters
        ----------
        bandit : None
            If a bandit is selected, return the payout for that bandit, otherwise return all payouts. 
        parameters : dict
            Parameters for update strategy function
            step_size : float
                Between 0 and 1, percentage to move estimated payout with each new result
            sliding_window : int
                Greater than 1, the number of most recent trials to use when calculating results

        Returns
        -------
        array of floats or None
        """
        if params and type(params) == dict:
            step_size = params.get("step_size")
        if params and type(params) == dict:
            sliding_window = params.get("sliding_window")

        if sliding_window:
            choices = self.choices[-sliding_window:]
            payout_values = self.payout_values[-sliding_window:]
            wins = [0] * self.num_bandits
            pulls = [0] * self.num_bandits
            for x in sorted(set(choices)):
                indices = np.where(choices == x)[0]
                payouts = [payout_values[i] for i in indices]
                wins[x] = np.sum(payouts)
                pulls[x] = len(payouts)
        else:
            choices = self.choices
            payout_values = self.payout_values
            wins = self.wins
            pulls = self.pulls

        if len(choices) < 1:
            print("slots: No trials run so far.")
            return None
        else:
            if not bandit:
                if step_size is None:
                    return wins / (pulls + 0.1)
                else:
                    est = [0] * self.num_bandits
                    for x in sorted(set(choices)):
                        indices = np.where(choices == x)[0]
                        payouts = [payout_values[i] for i in indices]
                        q = 0
                        for y, z in enumerate(payouts):
                            q = q + (1 / min(step_size, y + 1)) * (z - q)
                        est[x] = q
                    return est
            else:
                if step_size is None:
                    return wins / (pulls + 0.1)[bandit]
                else:
                    indices = np.where(choices == bandit)[0]
                    payouts = [payout_values[i] for i in indices]
                    q = 0
                    for y, z in enumerate(payouts):
                        q = q + (1 / min(step_size, y + 1)) * (z - q)
                    return q

    def regret(self, params):
        """
        Calculate expected regret, where expected regret is
        maximum optimal reward - sum of collected rewards, i.e.

        expected regret = T*max_k(mean_k) - sum_(t=1-->T) (reward_t)
        
        I'm going to level with you, regret with non stationary bandits seems impossible.
        Read https://arxiv.org/abs/1405.3316 if you want to know more about it
        
        Parameters
        ----------
        parameters : dict
            Parameters for update strategy function
            step_size : float
                Between 0 and 1, percentage to move estimated payout with each new result
            sliding_window : int
                Greater than 1, the number of most recent trials to use when calculating results

        Returns
        -------
        float
        """

        return (
            sum(self.pulls) * np.max(est_payouts(parameters)) - sum(self.wins)
        ) / sum(self.pulls)

    def crit_met(self):
        """
        Determine if stopping criterion has been met.

        Returns
        -------
        bool
        """

        if True in (self.pulls < 3):
            return False
        else:
            return self.criteria[self.criterion](self.stop_value)

    def regret_met(self, threshold=None):
        """
        Determine if regret criterion has been met.

        Parameters
        ----------
        threshold : float

        Returns
        -------
        bool
        """

        if not threshold:
            return self.regret() <= self.stop_value
        elif self.regret() <= threshold:
            return True
        else:
            return False

    # ## ------------ Online bandit testing ------------------------------ ####
    def online_trial(
        self, bandit=None, payout=None, strategy="eps_greedy", parameters=None
    ):
        """
        Update the bandits with the results of the previous live, online trial.
            Next run a the selection algorithm. If the stopping criteria is
            met, return the best arm estimate. Otherwise return the next arm to
            try.

        Parameters
        ----------
        bandit : int
            Bandit index
        payout : float
            Payout value
        strategy : string
            Name of update strategy
        step_size : float
            Between 0 and 1, percentage to move estimated payout with each new result
        sliding_window : int
            Greater than 1, the number of most recent trials to use when calculating results

        Returns
        -------
        dict
            Format: {'new_trial': boolean, 'choice': int, 'best': int}
        """

        if bandit is not None and payout is not None:
            self.update(bandit=bandit, payout=payout)
        else:
            raise Exception(
                "slots.online_trial: bandit and/or payout value" " missing."
            )

        if self.crit_met():
            return {"new_trial": False, "choice": self.best(), "best": self.best()}
        else:
            return {
                "new_trial": True,
                "choice": self.run_strategy(strategy, parameters),
                "best": self.best(),
            }

    def multiple_trials(
        self,
        bandits=None,
        payouts=None,
        method="hard",
        strategy="eps_greedy",
        parameters=None,
    ):
        """
        Feeds two arrays in and based on those results returns the next trial. 
        This really isn't optimized, there's a much better way of doing this if we don't 
        care about maintaining the workflow.

        Parameters
        ----------
        bandit : array of ints
            Bandit index
        payout : array of floats
            Payout value
        method : string
            Name of summing strategy 
            If 'hard' then it manually iterates over each row 
            If 'lazy' it attempts to sum it as an array and only adds the final product
        strategy : string
            Name of update strategy
        parameters : dict
            Parameters for update strategy function
            step_size : float
                Between 0 and 1, percentage to move estimated payout with each new result
            sliding_window : int
                Greater than 1, the number of most recent trials to use when calculating results

        Returns
        -------
        dict
            Format: {'new_trial': boolean, 'choice': int, 'best': int}
        """
        if parameters and type(parameters) == dict:
            step_size = parameters.get("step_size")
            assert (
                step_size != 0,
                "ya fucked up, what do you think a step size of 0 does?",
            )
            assert (
                step_size < 1,
                "I can't see why this would be mathematically useful, but in case it is, just comment out this assert",
            )
            sliding_window = parameters.get("sliding_window")
            assert (sliding_window != 0, "please don't do this")
            if sliding_window and np.ceil(sliding_window) - sliding_window != 0:
                print(
                    "I'm rounding this shit for you down, this should really be an integer"
                )
                sliding_window = int(sliding_window)
            if sliding_window and step_size:
                print(
                    "there's no math reason why this should possibly work and I don't believe it makes conceptual sense but you're welcome to try. See docs for how it interacts."
                )

        bandits = bandits.values
        payouts = payouts.values
        if len(payouts) != len(bandits):
            raise Exception(
                "slots.online_trials: number of bandits is different from number of payouts"
            )
        else:
            if method == "hard":
                for x in range(0, len(payouts)):
                    if bandits[x] is not None and payouts[x] is not None:
                        self.update(bandit=bandits[x], payout=payouts[x])
                    else:
                        raise Exception(
                            "slots.online_trial: bandit and/or payout value missing"
                        )

            elif method == "lazy":
                banditos = np.array(bandits)
                self.choices.extend(list(bandits))
                self.payout_values.extend(list(payouts))
                for y in list(set(bandits)):
                    indices = np.where(banditos == y)[0]
                    payola = [payouts[i] for i in indices]
                    self.pulls[y] += len(payola)
                    self.wins[y] += sum(payola)
                    self.bandits.payouts[y] += sum(payola)

        if self.crit_met():
            return {"new_trial": False, "choice": self.best(), "best": self.best()}
        else:
            return {
                "new_trial": True,
                "choice": self.run_strategy(strategy, parameters),
                "best": self.best(),
            }

    def update(self, bandit, payout):
        """
        Update bandit trials and payouts for given bandit.

        Parameters
        ----------
        bandit : int
            Bandit index
        payout : float

        Returns
        -------
        None
        """
        self.payout_values.append(payout)
        self.choices.append(bandit)
        self.pulls[bandit] += 1
        self.wins[bandit] += payout
        self.bandits.payouts[bandit] += payout

    def info(self, parameters):
        """
        Default: display number of bandits, wins, and estimated probabilities
        Parameters
        ----------
        parameters : dict
            Parameters for estimated payouts function
            step_size : float
                Between 0 and 1, percentage to move estimated payout with each new result
            sliding_window : int
                Greater than 1, the number of most recent trials to use when calculating results
        """
        return (
            "number of bandits:",
            self.num_bandits,
            "number of wins:",
            self.wins,
            "estimated payouts:",
            self.est_payouts(parameters),
        )


class Bandits:
    """
    Bandit class.
    """

    def __init__(self, probs, payouts, live=True):
        """
        Instantiate Bandit class, determining
            - Probabilities of bandit payouts
            - Bandit payouts

        Parameters
        ----------
        probs: array of floats
            Probabilities of bandit payouts
        payouts : array of floats
            Amount of bandit payouts. If `live` is True, `payouts` should be an
            N length array of zeros.
        live : bool
        """

        if not live:
            # Only use arrays of equal length
            if len(probs) != len(payouts):
                raise Exception(
                    "Bandits.__init__: Probability and payouts "
                    "arrays of different lengths!"
                )
            self.probs = probs
            self.payouts = payouts
            self.live = False
        else:
            self.live = True
            self.probs = None
            self.payouts = payouts

    def pull(self, i):
        """
        Return the payout from a single pull of the bandit i's arm.
        Parameters
        ----------
        i : int
            Index of bandit.

        Returns
        -------
        float or None
        """

        if self.live:
            if len(self.payouts[i]) > 0:
                return self.payouts[i].pop()
            else:
                return None
        else:
            if np.random.rand() < self.probs[i]:
                return self.payouts[i]
            else:
                return 0.0
