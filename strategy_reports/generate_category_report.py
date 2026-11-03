#!/usr/bin/env python3
"""生成策略类别分析 PDF 报告"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os, datetime

# ── 注册中文字体 ──────────────────────────────────────────────────────────────
FONT_PATHS = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode MS.ttf",
    "/System/Library/Fonts/PingFang.ttc",
]
FONT_NAME = "Chinese"
for fp in FONT_PATHS:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont(FONT_NAME, fp))
            break
        except Exception:
            continue
else:
    FONT_NAME = "Helvetica"  # fallback

# ── 样式 ──────────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    def s(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent],
                              fontName=FONT_NAME, **kw)
    return {
        "title":    s("title",   fontSize=22, leading=30, alignment=TA_CENTER,
                      textColor=colors.HexColor("#1a237e"), spaceAfter=6),
        "subtitle": s("sub",     fontSize=12, leading=18, alignment=TA_CENTER,
                      textColor=colors.HexColor("#555555"), spaceAfter=20),
        "h1":       s("h1",      fontSize=15, leading=22, spaceBefore=16,
                      spaceAfter=6, textColor=colors.HexColor("#283593"),
                      borderPadding=(0,0,2,0)),
        "h2":       s("h2",      fontSize=12, leading=18, spaceBefore=10,
                      spaceAfter=4, textColor=colors.HexColor("#1565c0")),
        "body":     s("body",    fontSize=10, leading=15, spaceAfter=4),
        "small":    s("small",   fontSize=9,  leading=13, spaceAfter=2,
                      textColor=colors.HexColor("#444444")),
        "caption":  s("caption", fontSize=9,  leading=12, alignment=TA_CENTER,
                      textColor=colors.HexColor("#666666")),
        "tag":      s("tag",     fontSize=10, leading=15,
                      textColor=colors.HexColor("#c62828")),
    }

# ── 数据 ──────────────────────────────────────────────────────────────────────
CATEGORIES = [
    {
        "name": "趋势跟踪",
        "en": "Trend Following",
        "count": 18,
        "pct": 25,
        "color": "#1565c0",
        "strategies": [
            "claude_dual_ma_cross", "claude_turtle_trading", "claude_adaptive_ma", "claude_ichimoku_cloud",
            "claude_supertrend", "claude_trend_hold", "claude_breakout_trailing", "claude_multi_timeframe_trend",
            "claude_donchian_breakout", "claude_parabolic_sar", "claude_heikin_ashi", "claude_ma_ribbon",
            "claude_trend_strength", "claude_triple_ema_cross", "claude_chandelier_exit",
            "claude_pullback_buy", "claude_atr_channel_trend", "claude_adx_trend_strength",
        ],
        "desc": "跟随价格趋势方向，利用均线、ATR通道、SAR等判断趋势方向，在趋势延续中持仓，趋势反转时离场。",
    },
    {
        "name": "均值回归",
        "en": "Mean Reversion",
        "count": 12,
        "pct": 17,
        "color": "#2e7d32",
        "strategies": [
            "claude_mean_reversion", "claude_zscore_reversion", "claude_connors_rsi", "claude_meanrev_bb_rsi",
            "claude_vwap_mean_reversion", "claude_adaptive_bollinger", "claude_combo_meanrev_plus",
            "claude_adaptive_rsi_meanrev", "claude_keltner_meanrev", "claude_mean_reversion_filtered",
            "claude_short_term_reversal", "claude_double_bottom_rsi",
        ],
        "desc": "价格偏离均值（布林带、VWAP、Z-Score通道）时逆向入场，等待回归中心价位后获利离场。",
    },
    {
        "name": "振荡器",
        "en": "Oscillator",
        "count": 8,
        "pct": 11,
        "color": "#6a1b9a",
        "strategies": [
            "claude_rsi_macd_trend", "claude_stoch_rsi", "claude_williams_r_adx", "claude_aroon_oscillator",
            "claude_stochastic_momentum", "claude_multi_osc_consensus", "claude_rsi_divergence",
            "claude_divergence_combo",
        ],
        "desc": "RSI、随机指标、Aroon等振荡器衡量超买超卖状态，结合背离信号捕捉短中期反转。",
    },
    {
        "name": "动量",
        "en": "Momentum",
        "count": 7,
        "pct": 10,
        "color": "#e65100",
        "strategies": [
            "claude_momentum_breakout", "claude_trix_momentum", "claude_coppock_curve",
            "claude_momentum_accel", "claude_momentum_quality", "claude_obv_momentum",
            "claude_vol_weighted_mom",
        ],
        "desc": "利用价格动量惯性，在动量加速阶段顺势追入，动量衰竭时离场，包含成交量加权版本。",
    },
    {
        "name": "波动率/通道突破",
        "en": "Volatility / Channel Breakout",
        "count": 7,
        "pct": 10,
        "color": "#00838f",
        "strategies": [
            "claude_volatility_breakout", "claude_keltner_breakout", "claude_squeeze_momentum",
            "claude_bb_squeeze_breakout", "claude_mass_index", "claude_range_expansion", "claude_dual_thrust",
        ],
        "desc": "监测波动率收缩（ATR/BB带宽极窄）后的扩张突破，利用能量积聚释放的方向性爆发。",
    },
    {
        "name": "成交量分析",
        "en": "Volume Analysis",
        "count": 7,
        "pct": 10,
        "color": "#4527a0",
        "strategies": [
            "claude_chaikin_money_flow", "claude_elder_force", "claude_acc_dist_trend",
            "claude_price_channel_vol", "claude_volume_climax", "claude_volume_profile", "claude_vwap_breakout",
        ],
        "desc": "通过OBV、AD线、CMF等量能指标分析资金流向，捕捉量价背离和成交量密集区支撑/阻力。",
    },
    {
        "name": "自适应/政权切换",
        "en": "Adaptive / Regime Switching",
        "count": 5,
        "pct": 7,
        "color": "#558b2f",
        "strategies": [
            "claude_adaptive_regime", "claude_volatility_regime", "claude_momentum_reversion_hybrid",
            "claude_adaptive_rsi_meanrev", "claude_adaptive_bollinger",
        ],
        "desc": "动态识别市场状态（趋势/震荡/高低波动率），在不同政权下切换不同子策略逻辑。",
    },
    {
        "name": "集成/投票",
        "en": "Ensemble / Voting",
        "count": 3,
        "pct": 4,
        "color": "#37474f",
        "strategies": [
            "claude_ensemble_voting", "claude_adaptive_ensemble_v2", "claude_multi_indicator",
        ],
        "desc": "聚合多个子策略的信号进行多数投票或动态加权，平滑单策略噪音，提高信号稳定性。",
    },
    {
        "name": "价格行为/形态",
        "en": "Price Action / Pattern",
        "count": 4,
        "pct": 6,
        "color": "#bf360c",
        "strategies": [
            "claude_ross_hook", "claude_gap_and_go", "claude_pivot_bounce", "claude_swing_failure",
        ],
        "desc": "识别1-2-3形态、缺口追涨、枢轴点反弹、假突破摆动失败等经典价格行为形态。",
    },
    {
        "name": "周期/频率分析",
        "en": "Cycle / Frequency Analysis",
        "count": 2,
        "pct": 3,
        "color": "#880e4f",
        "strategies": [
            "fft_cycle_trend (新增 ★)", "claude_ehlers_fisher",
        ],
        "desc": "FFT/Ehlers方法提取价格主导周期，在周期波谷买入波峰卖出，与趋势类策略正交。",
    },
    {
        "name": "Gemini 20 (find-new-str)",
        "en": "Gemini 20 Research Series",
        "count": 20,
        "pct": 22,
        "color": "#1a237e",
        "strategies": [
            "gemini_vortex_trend", "gemini_schaff_trend", "gemini_ehlers_cycle",
            "gemini_mcginley_dynamic", "gemini_hull_ma", "gemini_trend_persistence",
            "gemini_aroon_osc", "gemini_coppock", "gemini_keltner_meanrev",
            "gemini_zscore_spread", "gemini_bullish_engulfing", "gemini_morning_star",
            "gemini_hammer_bounce", "gemini_shooting_star", "gemini_monday_reversal",
            "gemini_monthly_window", "gemini_garch_filter", "gemini_anchored_vwap",
            "gemini_volume_sva", "gemini_pead_momentum",
        ],
        "desc": "由 Gemini (find-new-str) 研发的 20 个新策略，覆盖日历效应、GARCH波动率、高级K线形态及DSP周期分析，填补了之前报告中识别的空白领域。",
    },
]

GAPS = [
    {
        "name": "季节性/日历效应",
        "en": "Calendar / Seasonality",
        "priority": "高",
        "color": "#c62828",
        "desc": "月度效应（1月效应）、星期效应（周一低谷）、财报后漂移（PEAD）。学术验证充分，与现有策略完全正交。",
    },
    {
        "name": "GARCH波动率预测",
        "en": "GARCH Volatility Forecasting",
        "priority": "高",
        "color": "#c62828",
        "desc": "用GARCH(1,1)预测未来波动率，在高波动预期下缩仓/切换策略，低波动时加仓趋势策略。",
    },
    {
        "name": "K线形态识别",
        "en": "Candlestick Pattern",
        "priority": "中",
        "color": "#f57f17",
        "desc": "吞噬、晨星、锤子、十字星等经典K线形态配合成交量确认，纯价格行为类信号。",
    },
    {
        "name": "统计套利/协整",
        "en": "Statistical Arbitrage / Cointegration",
        "priority": "低",
        "color": "#558b2f",
        "desc": "单股场景下可用 ADF 协整检验价格与其自身历史的均值回归速度，或与ETF的spread。",
    },
]

def bar_table(categories, styles):
    """生成水平条形图表格"""
    data = [["类别", "策略数", "占比", "可视化"]]
    ts = [
        ("FONT", (0, 0), (-1, -1), FONT_NAME, 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#283593")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f5f5f5"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i, cat in enumerate(categories, 1):
        bar_len = max(1, int(cat["pct"] * 1.2))
        bar = "█" * bar_len
        data.append([
            cat["name"],
            str(cat["count"]),
            f"{cat['pct']}%",
            bar,
        ])
        ts.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor(cat["color"])))
        ts.append(("FONTSIZE", (3, i), (3, i), 8))
    t = Table(data, colWidths=[4.5*cm, 1.8*cm, 1.8*cm, 7.5*cm])
    t.setStyle(TableStyle(ts))
    return t

def strategy_table(strategies, color, styles):
    """策略名列表，3列"""
    rows = []
    row = []
    for s in strategies:
        row.append(Paragraph(f"• {s}", styles["small"]))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        while len(row) < 3:
            row.append(Paragraph("", styles["small"]))
        rows.append(row)
    t = Table(rows, colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), FONT_NAME, 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafafa")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(color)),
    ]))
    return t

def gap_table(gaps, styles):
    data = [["空白类别", "优先级", "说明"]]
    ts = [
        ("FONT", (0, 0), (-1, -1), FONT_NAME, 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b71c1c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#fff8f8"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i, g in enumerate(gaps, 1):
        data.append([g["name"], g["priority"], g["desc"]])
        p_color = g["color"]
        ts.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor(p_color)))
        ts.append(("FONTNAME", (1, i), (1, i), FONT_NAME))
    t = Table(data, colWidths=[4*cm, 1.8*cm, 10.8*cm])
    t.setStyle(TableStyle(ts))
    return t

# ── 主程序 ────────────────────────────────────────────────────────────────────
def build_pdf(out_path):
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
    )
    styles = make_styles()
    story = []

    # 封面区
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("AITrader 活跃策略分类分析报告", styles["title"]))
    story.append(Paragraph(
        f"策略总数：72 个活跃  |  生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                            color=colors.HexColor("#1a237e")))
    story.append(Spacer(1, 0.5*cm))

    # 总览汇总表
    story.append(Paragraph("一、策略类别总览", styles["h1"]))
    story.append(Paragraph(
        "下表汇总 72 个活跃策略按功能类别的分布情况：",
        styles["body"]
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(bar_table(CATEGORIES, styles))
    story.append(Spacer(1, 0.5*cm))

    # 各类别详情
    story.append(Paragraph("二、各类别详细策略列表", styles["h1"]))

    for cat in CATEGORIES:
        block = []
        header = (
            f'<font color="{cat["color"]}"><b>{cat["name"]}</b></font>'
            f'  <font color="#888888">({cat["en"]})</font>'
            f'  — {cat["count"]} 个策略，占比 {cat["pct"]}%'
        )
        block.append(Paragraph(header, styles["h2"]))
        block.append(Paragraph(cat["desc"], styles["small"]))
        block.append(Spacer(1, 0.15*cm))
        block.append(strategy_table(cat["strategies"], cat["color"], styles))
        block.append(Spacer(1, 0.3*cm))
        story.append(KeepTogether(block))

    # 空白类别分析
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("三、明显空白类别与建议扩充方向", styles["h1"]))
    story.append(Paragraph(
        "以下类别在现有 72 个活跃策略中几乎缺失，建议作为下一轮 /find-new-str 的目标：",
        styles["body"]
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(gap_table(GAPS, styles))
    story.append(Spacer(1, 0.5*cm))

    # 归档策略备注
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("四、归档策略（待重新验证）", styles["h1"]))
    archived_data = [
        ["策略名", "类别", "状态", "备注"],
        ["claude_macd_divergence", "背离", "归档", "原始策略，未经优化"],
        ["claude_kdj_dmi", "振荡器/趋势", "归档", "原始策略，未经优化"],
        ["claude_elder_triple_screen", "多周期综合", "归档", "原始策略，未经优化"],
        ["claude_linreg_trend", "统计/趋势", "归档", "第2轮研发，MYPOOL最终 19.6%"],
        ["claude_vol_adj_momentum", "动量", "归档", "第2轮研发，MYPOOL最终 37.0%"],
    ]
    at = Table(archived_data, colWidths=[4.5*cm, 3.5*cm, 2*cm, 6.6*cm])
    at.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), FONT_NAME, 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#546e7a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f5f5f5"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(at)
    story.append(Spacer(1, 0.5*cm))

    # 页脚说明
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#eeeeee")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "本报告由 Claude Code 自动生成 · AITrader /find-new-str 策略研发流水线",
        styles["caption"]
    ))

    doc.build(story)
    print(f"PDF 已生成: {out_path}")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_category_analysis.pdf")
    build_pdf(out)
