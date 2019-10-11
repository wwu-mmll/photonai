try:
    from smac.tae.execute_ta_run import ExecuteTARun
    from smac.configspace import Configuration
    __found__ = True
except:
    ExecuteTARun = object
    Configuration = object
    __found__ = False

class MyExecuteTARun(ExecuteTARun):

    def __init__(self, run_limit=100, ta=None, **kwargs):
        """Constructor
        Parameters
        ----------
        ta : list
            target algorithm command line as list of arguments
        runhistory: RunHistory
            runhistory to keep track of all runs; only used if set
        stats: Stats()
             stats object to collect statistics about runtime and so on
        run_obj: str
            run objective of SMAC
        par_factor: int
            penalization factor
        crash_cost : float
            cost that is used in case of crashed runs (including runs
            that returned NaN or inf)
        abort_on_first_run_crash: bool
            if true and first run crashes, raise FirstRunCrashedException
        """
        if __found__:
            super(MyExecuteTARun, self).__init__(ta, **kwargs)

            self.run_limit = run_limit
        else:
            raise ModuleNotFoundError("Module smac not found or not installed as expected. "
                                      "Please install the smac_requirements.txt PHOTON provides.")

    def start(self, config: Configuration,
              instance: str,
              cutoff: float = None,
              seed: int = 12345,
              instance_specific: str = "0",
              capped: bool = False):
        """Wrapper function for ExecuteTARun.run() to check configuration
        budget before the runs and to update stats after run
        Parameters
        ----------
            config : Configuration
                Mainly a dictionary param -> value
            instance : string
                Problem instance
            cutoff : float
                Runtime cutoff
            seed : int
                Random seed
            instance_specific: str
                Instance specific information (e.g., domain file or solution)
            capped: bool
                If true and status is StatusType.TIMEOUT,
                uses StatusType.CAPPED
        Returns
        -------
            status: enum of StatusType (int)
                {SUCCESS, TIMEOUT, CRASHED, ABORT}
            cost: float
                cost/regret/quality (float) (None, if not returned by TA)
            runtime: float
                runtime (None if not returned by TA)
            additional_info: dict
                all further additional run information
        """

        if self.runhistory.config_ids.get(config) and self.runhistory.get_runs_for_config(config):
            return None, None, 1, None
        else:
            raise NotImplementedError("PHOTONs seed management is not implemented yet. "
                                      "At this juncture we can not run a config multiple times correctly."
                                      "If your minR=maxR==1 and this error occurred some other things went wrong."
                                      "Please contact us.")
