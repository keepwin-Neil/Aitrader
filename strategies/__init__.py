"""
strategies package — public API unchanged from original strategies.py
"""
from typing import Optional

from .base import BaseStrategy
from .core import *
from .llm_openai import *
from .llm_gemini import *

from .core import (
    MultiIndicator, DualMaCross, MomentumBreakout, MeanReversion, TurtleTrading,
    VwapBreakout, AdaptiveMA, RsiMacdTrend, EnsembleVoting, IchimokuCloud,
    VolatilityBreakout, SuperTrendStrategy, RossHook, DivergenceCombo,
    TrendHold, MomentumReversionHybrid, BreakoutTrailing, MultiTimeframeTrend,
    AdaptiveRegime, ObvMomentum, KeltnerBreakout, ZScoreReversion,
    DonchianBreakout, WilliamsRAdx, ParabolicSarTrend, HeikinAshiTrend,
    StochasticMomentumIndex, ChaikinMoneyFlow, ElderForceIndex, TrixMomentum,
    ConnorsRsi, CoppockCurve, MassIndexReversal, AroonOscillator,
    SqueezeMomentum, AccDistTrend, EhlersFisher, VolumeWeightedMomentum,
    PriceChannelVolume, MeanRevBollingerRsi, DualThrust, RsiDivergence,
    MaRibbon, VwapMeanReversion, TrendStrengthAdxDi, GapAndGo,
    TripleEmaCross, MomentumAcceleration, PivotPointBounce, ChandelierExit,
    BbSqueezeBreakout, VolumeClimaxReversal, AdaptiveBollinger, PullbackBuy,
    DoubleBottomRsi, AtrChannelTrend, StochasticRsi, ComboMeanRevPlus,
    RangeContractionExpansion, MultiOscConsensus, AdaptiveRsiMeanRev,
    KeltnerMeanRev, VolumeProfileSupport, SwingFailure, VolatilityRegime,
    MomentumQuality, AdxTrendStrength, MeanReversionFiltered, ShortTermReversal,
    AdaptiveEnsembleV2, FftDominantCycle, EntropyMomentum,
    MacdDivergence, KdjDmi, ElderTripleScreen, LinearRegressionTrend,
    VolAdjMomentum, SeasonalityCalendar, GarchVolTiming, CandlePatternVol,
    HurstExponentStrategy, KaufmanEfficiencyBreakout,
)

from .llm_openai import (
    VhfRegimeBreakout, AdxDonchianThrust, VortexTrendPulse,
    RelativeVigorReversal, KstTrendSignal, SchaffCycleTrend, PpoTrendPulse,
    BalancePowerBreakout, CciTrendReclaim, DpoTrendReversal,
    AtrExpansionBreakout, RsiRangeShift, KeltnerMidlineResume,
    EmaSlopeBreakout,
)

from .llm_gemini import (
    GeminiVortexTrend, GeminiSchaffTrendCycle, GeminiEhlersCyberCycle,
    GeminiMcGinleyDynamic, GeminiHullMaCross, GeminiTrendPersistence,
    GeminiAroonOscillator, GeminiCoppockMomentum, GeminiKeltnerReversion,
    GeminiZScoreSpread, GeminiBullishEngulfingPlus, GeminiMorningStarPlus,
    GeminiHammerBounce, GeminiShootingStarReversal, GeminiMondayReversal,
    GeminiMonthlyWindow, GeminiGarchTrendFilter, GeminiVwapAnchoredTrend,
    GeminiVolumeProfileSva, GeminiPeadMomentum,
    GeminiFractalTrend, GeminiFisherTransform, GeminiKlingerVolume,
    GeminiSuperTrendV2, GeminiChandeMomentum, GeminiAdaptiveAlpha,
    GeminiPriceActionLiquidity, GeminiRegimeSwitching, GeminiFractalDimension,
    GeminiVolatilityBurst, GeminiMomentumExhaustion, GeminiStochasticRsiBurst,
    GeminiVolumePriceDiv, GeminiTripleScreenPulse, GeminiSupportResistanceSva,
    GeminiMacdHistDiv, GeminiBollingerImpulse, GeminiHeikinAshiBalance,
    GeminiAtrVolatilityStop, GeminiStdDevChannel,
    OpenAiTsiTrendline, OpenAiDemaReclaim, OpenAiTemaImpulse, OpenAiVwmaBreakout,
    OpenAiEnvelopePullback, OpenAiAtrRatioTrend, OpenAiUltimateBreakout,
    OpenAiRsiFailureSwing, OpenAiMfiPressurePop, OpenAiRocTrendShift,
    OpenAiDisparityRebound, OpenAiPercentBTrend, OpenAiBandwidthExpansion,
    OpenAiKeltnerBandPulse, OpenAiZscoreTrendSnap,
    OpenAiIntradayIntensityBreakout, OpenAiEaseMoveTrend, OpenAiCmfReclaim,
    OpenAiForceTrendPop, OpenAiVolumeEfficiencyTrend, OpenAiInsideBarThrust,
    OpenAiWideRangeReversal, OpenAiGapReclaim, OpenAiThreeBarStrength,
    OpenAiHammerTrendConfirm, OpenAiChaikinVolBreak, OpenAiMassIndexPop,
    OpenAiRangeCompressionPulse, OpenAiTrueRangePop, OpenAiDonchianMidReclaim,
)


STRATEGY_REGISTRY: dict[str, BaseStrategy] = {}

# 已归档策略仓库（代码保留，不参与回测）
STRATEGY_WAREHOUSE: dict[str, BaseStrategy] = {}


def _register(*classes):
    for cls in classes:
        s = cls()
        STRATEGY_REGISTRY[s.name] = s


def _archive(*classes):
    """将策略放入仓库，不参与正常回测"""
    for cls in classes:
        s = cls()
        STRATEGY_WAREHOUSE[s.name] = s


# 活跃策略（71个）
_register(MultiIndicator, DualMaCross, MomentumBreakout,
          MeanReversion, TurtleTrading,
          VwapBreakout, AdaptiveMA, RsiMacdTrend, EnsembleVoting,
          IchimokuCloud, VolatilityBreakout, SuperTrendStrategy,
          RossHook, DivergenceCombo,
          TrendHold, MomentumReversionHybrid, BreakoutTrailing,
          MultiTimeframeTrend, AdaptiveRegime,
          ObvMomentum, KeltnerBreakout, ZScoreReversion,
          DonchianBreakout, WilliamsRAdx, ParabolicSarTrend, HeikinAshiTrend,
          StochasticMomentumIndex, ChaikinMoneyFlow, ElderForceIndex,
          TrixMomentum, ConnorsRsi, CoppockCurve, MassIndexReversal,
          AroonOscillator, SqueezeMomentum,
          AccDistTrend, EhlersFisher, VolumeWeightedMomentum,
          PriceChannelVolume, MeanRevBollingerRsi, DualThrust,
          RsiDivergence, MaRibbon,
          VwapMeanReversion, TrendStrengthAdxDi, GapAndGo,
          TripleEmaCross, MomentumAcceleration, PivotPointBounce,
          ChandelierExit,
          BbSqueezeBreakout, VolumeClimaxReversal, AdaptiveBollinger,
          PullbackBuy,
          DoubleBottomRsi, AtrChannelTrend, StochasticRsi, ComboMeanRevPlus,
          RangeContractionExpansion, MultiOscConsensus,
          AdaptiveRsiMeanRev, KeltnerMeanRev, VolumeProfileSupport,
          SwingFailure,
          VolatilityRegime, MomentumQuality, AdxTrendStrength,
          MeanReversionFiltered, ShortTermReversal, AdaptiveEnsembleV2,
          FftDominantCycle,   # 新晋活跃策略: MYPOOL验收 71.66%
          EntropyMomentum,
          # Gemini 20 策略
          GeminiVortexTrend, GeminiSchaffTrendCycle, GeminiEhlersCyberCycle,
          GeminiMcGinleyDynamic, GeminiHullMaCross, GeminiTrendPersistence,
          GeminiAroonOscillator, GeminiCoppockMomentum, GeminiKeltnerReversion,
          GeminiZScoreSpread, GeminiBullishEngulfingPlus, GeminiMorningStarPlus,
          GeminiHammerBounce, GeminiShootingStarReversal, GeminiMondayReversal,
          GeminiMonthlyWindow, GeminiGarchTrendFilter, GeminiVwapAnchoredTrend,
          GeminiVolumeProfileSva, GeminiPeadMomentum)

# 归档策略（含候选/待验证策略）
_archive(MacdDivergence, KdjDmi, ElderTripleScreen,
         LinearRegressionTrend, VolAdjMomentum,
         SeasonalityCalendar, GarchVolTiming, CandlePatternVol,
         HurstExponentStrategy, KaufmanEfficiencyBreakout,
         GeminiFractalTrend, GeminiFisherTransform, GeminiKlingerVolume,
         GeminiSuperTrendV2, GeminiChandeMomentum, GeminiAdaptiveAlpha,
         GeminiPriceActionLiquidity, GeminiRegimeSwitching,
         GeminiFractalDimension, GeminiVolatilityBurst,
         GeminiMomentumExhaustion, GeminiStochasticRsiBurst,
         GeminiVolumePriceDiv, GeminiTripleScreenPulse,
         GeminiSupportResistanceSva, GeminiMacdHistDiv,
         GeminiBollingerImpulse, GeminiHeikinAshiBalance,
         GeminiAtrVolatilityStop, GeminiStdDevChannel,
         VortexTrendPulse, RelativeVigorReversal, KstTrendSignal,
         SchaffCycleTrend, PpoTrendPulse, BalancePowerBreakout,
         CciTrendReclaim, DpoTrendReversal, AtrExpansionBreakout,
         RsiRangeShift, KeltnerMidlineResume,
         VhfRegimeBreakout, AdxDonchianThrust,
         EmaSlopeBreakout, OpenAiTsiTrendline, OpenAiDemaReclaim,
         OpenAiTemaImpulse, OpenAiVwmaBreakout, OpenAiEnvelopePullback,
         OpenAiAtrRatioTrend, OpenAiUltimateBreakout, OpenAiRsiFailureSwing,
         OpenAiMfiPressurePop, OpenAiRocTrendShift, OpenAiDisparityRebound,
         OpenAiPercentBTrend, OpenAiBandwidthExpansion, OpenAiKeltnerBandPulse,
         OpenAiZscoreTrendSnap, OpenAiIntradayIntensityBreakout,
         OpenAiEaseMoveTrend, OpenAiCmfReclaim, OpenAiForceTrendPop,
         OpenAiVolumeEfficiencyTrend, OpenAiInsideBarThrust,
         OpenAiWideRangeReversal, OpenAiGapReclaim, OpenAiThreeBarStrength,
         OpenAiHammerTrendConfirm, OpenAiChaikinVolBreak, OpenAiMassIndexPop,
         OpenAiRangeCompressionPulse, OpenAiTrueRangePop,
         OpenAiDonchianMidReclaim)


def _resolve_strategy_alias(name: str) -> Optional[str]:
    """Resolve public short names like ``trix_momentum`` to one unique owner-prefixed strategy."""
    if not name or "_" not in name:
        return None
    all_names = [*STRATEGY_REGISTRY.keys(), *STRATEGY_WAREHOUSE.keys()]
    matches = [item for item in all_names if item.endswith(f"_{name}")]
    if len(matches) == 1:
        return matches[0]
    return None


def get_strategy(name: str) -> BaseStrategy:
    resolved = name
    if resolved not in STRATEGY_REGISTRY and resolved not in STRATEGY_WAREHOUSE:
        resolved = _resolve_strategy_alias(name) or name
    if resolved in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[resolved]
    if resolved in STRATEGY_WAREHOUSE:
        return STRATEGY_WAREHOUSE[resolved]
    raise ValueError(f"未知策略: {name}，可选: {list(STRATEGY_REGISTRY.keys())}")


def list_strategies() -> list[dict]:
    return [{"name": s.name, "description": s.description,
             "default_params": s.default_params}
            for s in STRATEGY_REGISTRY.values()]


def list_warehouse() -> list[dict]:
    """列出仓库中归档的策略"""
    return [{"name": s.name, "description": s.description,
             "default_params": s.default_params, "status": "archived"}
            for s in STRATEGY_WAREHOUSE.values()]
