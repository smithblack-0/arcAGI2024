"""
Centrally located config object.

I figure if I ever feel like it I can make this load from
a proper config and just display the right properties
"""

from enum import Enum

class Verbosity(Enum):
    Debug = 3
    Anomaly = 2
    NonterminalError = 1
    TerminalError = 0



class SchemaConfig(Enum):


class Config(Enum):
    """
    The config object. Contains a bunch of config info
    """
    # Mode names and Mode feature
    STOP_GEN = "Stop"
    TEXT_MODE = "Text"
    GRIDMODE = "GridInt"
    MODES = [STOP_GEN, TEXT_MODE, GRIDMODE]

    # Zone names and Zone features

    RULEGEN = "RuleStatement"
    SCENARIOGEN = "Scenario"
    SCENARIOANSWER = "ScenarioAnswer"
    RULESDEDUCTIONSTEPS = "RulesDeductionSteps"
    RULESDEDUCEDANSWER = "RulesDeductedAnswer"
    SOLUTIONSTEPS = "SolutionSteps"
    SOLUTIONANSWER = "Solution"
    ZONES = [RULEGEN,
             SCENARIOGEN,
             SCENARIOANSWER,
             RULESDEDUCTIONSTEPS,
             RULESDEDUCEDANSWER,
             SOLUTIONSTEPS,
             SOLUTIONANSWER]

    ##
    # Allowed access per zone. Very important to avoid leaking information
    # where it is not allowed.
    ##

    ZONE_ACCESS = { RULEGEN : [],
                    SCENARIOGEN : [RULEGEN],
                    SCENARIOANSWER : [RULEGEN, SCENARIOGEN],
                    RULESDEDUCTIONSTEPS: [SCENARIOGEN],
                    RULESDEDUCEDANSWER: [SCENARIOGEN, RULESDEDUCTIONSTEPS],
                    SOLUTIONSTEPS: [SCENARIOGEN, RULESDEDUCEDANSWER],
                    SOLUTIONANSWER: [SCENARIOGEN, RULESDEDUCEDANSWER, SOLUTIONSTEPS],
    }

    ##
    # Automated forcing per zone.
    ##

    ZONE_FORCING = {RULESDEDUCEDANSWER : RULEGEN,
                    SOLUTIONANSWER: SCENARIOANSWER}

    ##
    # Shapes for the modes
    ##

    SHAPES = {STOP_GEN : [], }