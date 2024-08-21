"""Module for global flags
"""

from prism import Flag, global_flag


@global_flag
class FlagTrade(Flag):
    # Including Trade setting to include/exclude interregional fuel trade:
    # FlagTrade='endogenous': endogenous trade, FlagTrade='exogenous': exogenous trade
    default = 'endogenous'
    options = ['endogenous', 'exogenous']


@global_flag
class FlagTaxBeforeTrade(Flag):
    # 'after' = apply carbon tax on fossil losses after trade,
    # 'before = apply tax on fossil losses before trade (carbon tax on direct emissions
    #           is always applied after trade)
    default = 'after'
    options = ['after', 'before']


@global_flag
class FlagForesight(Flag):
    # 'no foresight' = no foresight, use current carbon price in epg/DAC choice of technology
    # 'history' = history repeating expectation of carbon price, increase of last 20 years projected
    # 'perfect' = perfect foresight, carbon price is that of ForesightYears years forward
    default = 'perfect'
    options = ['no foresight', 'history', 'perfect']


@global_flag
class FlagLearning(Flag):
    # 'normal' = Normal,
    # 'baseline' = Baseline (no policy induced),
    # 'no learning' = no learning
    default = 'normal'
    options = ['normal', 'baseline', 'no learning']


class FlagEndoEROI(Flag):
    # 'endogenous' = endogenous EROI calculation,
    # 'exogenous' = exogenous EROI calculations
    default = 'exogenous'
    options = ['exogenous', 'endogenous']


@global_flag
class FlagCTaxFossilExport(Flag):
    # 'import' = apply import region ctax on losses
    # 'export' = Apply export region ctax on losses
    # (carbon tax from direct emissions is always applied in region of fossil use)
    default = 'import'
    options = ['import', 'export']
