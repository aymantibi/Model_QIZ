#!/usr/bin/env python3
"""
Generate QIZ Model Explainer PDF using ReportLab.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
import os

OUTPUT = r"C:\Users\Admin\Desktop\Idea QIZs and Development\Model\qiz_model_explainer.pdf"

# ── Styles ──────────────────────────────────────────────────────────────────

styles = getSampleStyleSheet()

TITLE = ParagraphStyle("DocTitle",
    parent=styles["Title"],
    fontSize=22, leading=28, spaceAfter=6, alignment=TA_CENTER,
    textColor=colors.HexColor("#1a1a2e"))

SUBTITLE = ParagraphStyle("DocSubtitle",
    parent=styles["Normal"],
    fontSize=13, leading=18, spaceAfter=4, alignment=TA_CENTER,
    textColor=colors.HexColor("#4a4a6a"), fontName="Helvetica-Oblique")

DATE_STYLE = ParagraphStyle("DocDate",
    parent=styles["Normal"],
    fontSize=10, leading=14, spaceAfter=2, alignment=TA_CENTER,
    textColor=colors.HexColor("#888888"))

H1 = ParagraphStyle("H1",
    parent=styles["Heading1"],
    fontSize=15, leading=20, spaceBefore=18, spaceAfter=6,
    textColor=colors.HexColor("#1a1a2e"),
    fontName="Helvetica-Bold",
    borderPad=4)

H2 = ParagraphStyle("H2",
    parent=styles["Heading2"],
    fontSize=12, leading=16, spaceBefore=12, spaceAfter=4,
    textColor=colors.HexColor("#2e4a6e"),
    fontName="Helvetica-Bold")

BODY = ParagraphStyle("Body",
    parent=styles["Normal"],
    fontSize=10.5, leading=16, spaceAfter=6,
    alignment=TA_JUSTIFY,
    fontName="Helvetica")

EQ = ParagraphStyle("Equation",
    parent=styles["Normal"],
    fontSize=10.5, leading=16,
    leftIndent=36, rightIndent=36,
    spaceBefore=6, spaceAfter=6,
    fontName="Helvetica-Oblique",
    textColor=colors.HexColor("#1a1a2e"))

EQ_LABEL = ParagraphStyle("EqLabel",
    parent=EQ,
    rightIndent=0,
    alignment=TA_LEFT)

CODE = ParagraphStyle("Code",
    parent=styles["Normal"],
    fontSize=8.5, leading=13,
    leftIndent=24, rightIndent=24,
    spaceBefore=4, spaceAfter=4,
    fontName="Courier",
    backColor=colors.HexColor("#f4f4f8"),
    textColor=colors.HexColor("#1a1a2e"))

NOTE = ParagraphStyle("Note",
    parent=styles["Normal"],
    fontSize=9.5, leading=14,
    leftIndent=18, rightIndent=18,
    spaceBefore=4, spaceAfter=4,
    fontName="Helvetica-Oblique",
    textColor=colors.HexColor("#444466"))

BULLET = ParagraphStyle("Bullet",
    parent=BODY,
    leftIndent=20, bulletIndent=8,
    spaceBefore=2, spaceAfter=2)

BUG_HIGH = ParagraphStyle("BugHigh",
    parent=BODY,
    leftIndent=14,
    borderColor=colors.HexColor("#c0392b"),
    borderWidth=2, borderPad=6,
    backColor=colors.HexColor("#fdf0ee"))

BUG_MED = ParagraphStyle("BugMed",
    parent=BODY,
    leftIndent=14,
    borderColor=colors.HexColor("#e67e22"),
    borderWidth=2, borderPad=6,
    backColor=colors.HexColor("#fef9f0"))

BUG_LOW = ParagraphStyle("BugLow",
    parent=BODY,
    leftIndent=14,
    borderColor=colors.HexColor("#27ae60"),
    borderWidth=2, borderPad=6,
    backColor=colors.HexColor("#f0fdf4"))

# ── Helpers ──────────────────────────────────────────────────────────────────

def h1(text):
    return Paragraph(text, H1)

def h2(text):
    return Paragraph(text, H2)

def p(text):
    return Paragraph(text, BODY)

def eq(text):
    return Paragraph(text, EQ)

def code(text):
    # escape for XML
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(text, CODE)

def note(text):
    return Paragraph(f"<i>Note: {text}</i>", NOTE)

def sp(h=6):
    return Spacer(1, h)

def rule():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#ccccdd"), spaceAfter=4, spaceBefore=4)

def bullet(text):
    return Paragraph(f"• &nbsp; {text}", BULLET)

def sub_bullet(text):
    return Paragraph(f"&nbsp;&nbsp;&nbsp;– {text}", BULLET)

def build_table(headers, rows, col_widths=None):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 9),
        ("BOTTOMPADDING", (0,0), (-1,0), 7),
        ("TOPPADDING",    (0,0), (-1,0), 7),
        ("FONTNAME",   (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",   (0,1), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
            [colors.HexColor("#f7f7fc"), colors.white]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#ccccdd")),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,1), (-1,-1), 5),
        ("BOTTOMPADDING",(0,1), (-1,-1), 5),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t

# ── Page template ─────────────────────────────────────────────────────────────

def on_page(canvas, doc):
    canvas.saveState()
    # footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(2*cm, 1.2*cm,
        "QIZ Heterogeneous-Firm Trade Model — Personal Study Guide")
    canvas.drawRightString(A4[0]-2*cm, 1.2*cm, f"Page {doc.page}")
    # top rule
    canvas.setStrokeColor(colors.HexColor("#1a1a2e"))
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, A4[1]-1.8*cm, A4[0]-2*cm, A4[1]-1.8*cm)
    canvas.restoreState()

# ── Document ──────────────────────────────────────────────────────────────────

def build():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm,  bottomMargin=2.5*cm,
        title="QIZ Heterogeneous-Firm Trade Model — Study Guide",
        author="Model Review",
    )

    story = []

    # ── Title page ────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 2*cm),
        Paragraph("Understanding the QIZ", TITLE),
        Paragraph("Heterogeneous-Firm Trade Model", TITLE),
        sp(8),
        Paragraph("A Personal Study Guide", SUBTITLE),
        Paragraph("Two Regions · Multi-Sector · Endogenous Compliance & Upgrading", SUBTITLE),
        sp(6),
        Paragraph("February 2026", DATE_STYLE),
        Spacer(1, 1.5*cm),
        rule(),
        sp(10),
        p("This document explains the QIZ heterogeneous-firm trade model in full detail. "
          "It covers the economic theory behind each block — demand, firm costs, trade, "
          "compliance, upgrading — and ties each concept to the Python implementation. "
          "All key derivations (unit costs, price indices, markups) are worked through "
          "step by step. A final section documents the bugs found in the code and the "
          "corrections applied."),
        PageBreak(),
    ]

    # ── Section 1: Overview ───────────────────────────────────────────────────
    story += [
        h1("1.  Overview"),
        rule(),
        h2("1.1  What is this model trying to do?"),
        p("The model asks: <b>what happens to Egyptian firms and workers when Egypt gains "
          "preferential access to the US market through Qualifying Industrial Zones (QIZs)?</b>"),
        p("QIZs are special industrial zones in Egypt (and Jordan) where firms can export "
          "to the United States <b>tariff-free</b>, provided their products contain a minimum "
          "share of Israeli inputs. This is the <b>Rules of Origin (ROO)</b> requirement. "
          "The model captures three central firm-level responses:"),
        bullet("Which firms choose to <b>comply</b> with the ROO (and thus export tariff-free to the US)."),
        bullet("Which firms <b>upgrade their productivity</b> because they gain US market access."),
        bullet("How <b>wages, employment, and welfare</b> in Egypt change in equilibrium."),
        sp(8),

        h2("1.2  The three-layer structure"),
        p("The model is solved as three nested loops:"),
        sp(4),
        code(
            "OUTER LOOP  (General Equilibrium)\n"
            "  Solves for: wages {w_Q, w_N} and entry masses {M_rs}\n"
            "\n"
            "  GOODS BLOCK  (Price indices & expenditure)\n"
            "    Solves for: domestic sector price indices {P_EG,s}\n"
            "    Computes: labor allocation, Egyptian income,\n"
            "              sector expenditures, price indices\n"
            "\n"
            "    FIRM PROBLEM  (innermost)\n"
            "      For every (phi, eps) draw: choose\n"
            "        - Which markets to serve  (EG, US, RW)\n"
            "        - Whether to comply with ROO  (Q firms only)\n"
            "        - Whether to upgrade productivity"
        ),
        sp(8),

        h2("1.3  Regions, sectors, and destinations"),
        sp(4),
        build_table(
            ["Symbol", "Meaning"],
            [
                ["Q",   "QIZ-eligible region"],
                ["N",   "Non-QIZ region"],
                ["T",   "Textiles / Apparel sector"],
                ["O",   "Other manufacturing sector"],
                ["EG",  "Domestic Egypt market"],
                ["US",  "United States market"],
                ["RW",  "Rest of World market"],
            ],
            col_widths=[3*cm, 11*cm],
        ),
        sp(6),
        p("Egypt is modeled as a <b>small open economy</b>: it takes foreign prices and "
          "expenditures as given. What happens in Egypt does not move US or RW price indices."),
        PageBreak(),
    ]

    # ── Section 2: Demand & Price Indices ────────────────────────────────────
    story += [
        h1("2.  Demand and Price Indices"),
        rule(),
        h2("2.1  The consumer's problem — nested CES"),
        p("Consumers in Egypt (and in the US and RW) have <b>nested CES preferences</b> — "
          "a two-level structure. The upper tier governs how spending is split across "
          "sectors; the lower tier governs substitution across varieties within a sector."),
        sp(6),

        h2("2.2  Upper tier — across-sector substitution"),
        p("The upper-tier utility aggregator is:"),
        eq("U_j  =  [ Σ_s  β_s^(1/η)  ·  U_{js}^((η−1)/η) ]^(η/(η−1))"),
        p("where:"),
        bullet("<i>β_s</i> > 0 are sector expenditure-share weights  (Σ_s β_s = 1)"),
        bullet("<i>η</i> > 0 is the elasticity of substitution <i>across</i> sectors"),
        bullet("<i>U_{js}</i> is the within-sector utility aggregate (defined below)"),
        p("Think of this as a basket: the consumer can substitute between 'textiles' and "
          "'other goods' with elasticity η. In the code: η = 2.0, β_T = 0.45, β_O = 0.55."),
        sp(6),

        h2("2.3  Lower tier — within-sector (Dixit-Stiglitz)"),
        p("Within each sector <i>s</i>, consumers value variety (Dixit-Stiglitz aggregator):"),
        eq("U_{js}  =  [ ∫_{ω ∈ Ω_{js}}  q_{js}(ω)^((σ_s−1)/σ_s)  dω ]^(σ_s/(σ_s−1))"),
        p("Here σ_s > 1 is the <b>elasticity of substitution between varieties</b> within "
          "sector <i>s</i>. A higher σ_s means varieties are closer substitutes — consumers "
          "are more price-sensitive. In the code: σ_T = 6.7, σ_O = 4.0."),
        sp(6),

        h2("2.4  Deriving individual variety demand"),
        p("To find how much of variety ω is demanded, maximize the lower-tier utility "
          "subject to a sector budget E_{js}. The first-order condition gives:"),
        eq("q_{js}(ω)  =  ( p_{js}(ω) / P_{js} )^(−σ_s)  ·  ( E_{js} / P_{js} )"),
        p("<b>Intuition:</b> demand falls when ω's own price rises (own-price elasticity −σ_s) "
          "and falls when the sector price index P_{js} falls (other varieties become cheaper)."),
        sp(6),

        h2("2.5  Deriving the CES price index"),
        p("The price index P_{js} is defined as the <b>minimum expenditure needed to "
          "attain one unit of sector-s utility</b>. Substituting optimal demands back "
          "into the budget constraint and simplifying yields:"),
        eq("P_{js}  =  [ ∫_{ω ∈ Ω_{js}}  p_{js}(ω)^(1−σ_s)  dω ]^(1/(1−σ_s))"),
        p("<b>Key properties:</b>"),
        bullet("Since 1 − σ_s < 0, each variety's contribution p^(1−σ_s) is "
               "<i>smaller</i> for higher-priced varieties — expensive varieties "
               "contribute less to the index."),
        bullet("P_{js} <i>falls</i> when more varieties enter (more competition lowers "
               "the cost of living) or when existing varieties cut prices."),
        p("In the code, the price index is built numerically by summing p^(1−σ) "
          "contributions across all active firms, then raising to the power 1/(1−σ):"),
        code(
            "# Accumulate p^{1-sigma} from each active firm\n"
            "contrib += wt * (pEG ** (1.0 - sigma))\n"
            "\n"
            "# Then convert: P = (sum of p^{1-sigma})^{1/(1-sigma)}\n"
            "P_new[s] = ces_price_from_power(P_pow, sigma)"
        ),
        sp(6),

        h2("2.6  Revenue of a variety"),
        p("Revenue = price × quantity:"),
        eq("R_{js}(ω)  =  E_{js} · ( p_{js}(ω) / P_{js} )^(1−σ_s)"),
        p("Since 1 − σ_s < 0, a <b>higher price → lower revenue share</b>. "
          "This is the standard CES demand system."),
        sp(6),

        h2("2.7  Aggregate price index across sectors"),
        eq("P_EG  =  [ Σ_s  β_s · P_{EG,s}^(1−η) ]^(1/(1−η))"),
        p("The special case η = 1 (Cobb-Douglas across sectors) is handled "
          "separately using the log formula: P = Π_s  P_s^(β_s)."),
        sp(6),

        h2("2.8  Egyptian income and sector expenditures"),
        p("Egyptian income:"),
        eq("Y_EG  =  w_Q · L_Q  +  w_N · L_N  +  T"),
        p("where <i>T</i> is a net transfer that closes the small-open economy "
          "(set to zero in the baseline). Upper-tier CES demand gives sector expenditures:"),
        eq("E_{EG,s}  =  β_s · ( P_{EG,s} / P_EG )^(1−η) · Y_EG"),
        p("<b>Intuition:</b> if sector <i>s</i> gets relatively cheaper (its price "
          "index falls), the expenditure share on that sector rises — when η > 1, "
          "sectors are substitutes."),
        PageBreak(),
    ]

    # ── Section 3: Firm Technology & Unit Costs ───────────────────────────────
    story += [
        h1("3.  Firm Technology and Unit Costs"),
        rule(),
        h2("3.1  Production function"),
        p("Each firm produces using labor <i>l</i> and an intermediate input bundle <i>m</i>:"),
        eq("y  =  φ · l^(α_s) · m^(1−α_s)"),
        p("where:"),
        bullet("<i>φ</i> is the firm's <b>productivity</b> (higher = more output per input)"),
        bullet("<i>α_s</i> ∈ (0,1) is the labor share in production"),
        bullet("1 − α_s is the intermediate input share"),
        p("In the code: α_T = 0.35, α_O = 0.30."),
        sp(6),

        h2("3.2  Deriving marginal cost — step by step"),
        p("The firm minimizes total cost  w_r · l + c_m · m  subject to producing y units."),
        p("<b>Step 1 — input demand ratio.</b> The first-order conditions give:"),
        eq("l / m  =  (α_s / (1−α_s)) · (c_m / w_r)"),
        p("<b>Step 2 — solve for inputs in terms of output.</b> Substituting back into "
          "the production function:"),
        eq("l  =  (y / φ) · (α_s / (1−α_s))^(1−α_s) · (c_m / w_r)^(1−α_s)"),
        eq("m  =  (y / φ) · (α_s / (1−α_s))^(−α_s) · (c_m / w_r)^(−α_s)     [after simplification]"),
        p("<b>Step 3 — total cost.</b> Substituting both into w_r·l + c_m·m:"),
        eq("C(y)  =  (y / φ) · [ w_r^(α_s) · c_m^(1−α_s) ] / [ α_s^(α_s) · (1−α_s)^(1−α_s) ]"),
        p("The denominator  α_s^(α_s) · (1−α_s)^(1−α_s)  is a constant that comes "
          "from the algebra of Cobb-Douglas cost minimization."),
        p("<b>Step 4 — marginal cost.</b> Since cost is linear in output, MC = AC:"),
        eq("mc(φ; c_m)  =  (1/φ) · w_r^(α_s) · c_m^(1−α_s)  /  [ α_s^(α_s) · (1−α_s)^(1−α_s) ]"),
        p("In the code:"),
        code(
            "alpha = p['alpha'][s]\n"
            "denom = (alpha ** alpha) * ((1.0 - alpha) ** (1.0 - alpha))\n"
            "mc_base = (1.0 / (phi * delta)) * (w_r**alpha * cN**(1-alpha)) / denom"
        ),
        p("Here <i>cN = p_rw[s]</i> is the freely-sourced intermediate price and "
          "<i>delta</i> is the upgrading multiplier (1 if no upgrade). "
          "Notice that <b>mc ∝ 1/φ</b>: more productive firms have lower marginal cost."),
        sp(6),

        h2("3.3  Markup pricing — derivation"),
        p("Under monopolistic competition with CES demand, every firm charges a "
          "<b>constant markup</b> over marginal cost. To see why, the firm faces "
          "demand  q = A · p^(−σ_s)  where A collects all terms outside the firm's control. "
          "Revenue is  R = p · q = A · p^(1−σ_s). Profit is:"),
        eq("π  =  R − mc · q  =  A · [ p^(1−σ_s) − mc · p^(−σ_s) ]"),
        p("Setting dπ/dp = 0:"),
        eq("A(1−σ_s) p^(−σ_s)  +  A σ_s · mc · p^(−σ_s−1)  =  0"),
        p("Solving for <i>p</i>:"),
        eq("p  =  σ_s / (σ_s − 1) · mc  =  μ_s · mc"),
        p("The <b>markup</b>  μ_s = σ_s / (σ_s − 1)  is higher when σ_s is lower "
          "(less substitutable varieties → more market power):"),
        bullet("Textiles: σ_T = 6.7  →  μ_T = 6.7/5.7 ≈ 1.175  (17.5% markup)"),
        bullet("Other mfg: σ_O = 4.0  →  μ_O = 4/3 ≈ 1.333  (33.3% markup)"),
        sp(6),

        h2("3.4  Variable profit"),
        p("A key CES identity: for any destination j, variable profit is always "
          "a fixed fraction of revenue:"),
        eq("π^var_{rjs}  =  R_{rjs} − mc · q_{rjs}  =  R_{rjs} − R_{rjs}/μ_s  =  R_{rjs} / σ_s"),
        p("This is why the code uses  pi_EG = R_EG / sigma,  pi_US = R_US / sigma,  etc."),
        PageBreak(),
    ]

    # ── Section 4: Trade Costs, Tariffs & Pricing ─────────────────────────────
    story += [
        h1("4.  Trade Costs, Tariffs, and Pricing"),
        rule(),
        h2("4.1  Iceberg trade costs"),
        p("Shipping goods to destination j from region r incurs an <b>iceberg cost</b> "
          "d_{rjs} ≥ 1: to deliver 1 unit, the firm must ship d_{rjs} units. The extra "
          "d_{rjs} − 1 units 'melt' in transit. This scales the effective marginal cost "
          "of delivery to:"),
        eq("Effective delivery cost  =  d_{rjs} · mc_{rs}(φ)"),
        p("In the baseline: d_{Q,US,s} = 1.35, d_{N,US,s} = 1.50 "
          "(QIZ firms have lower US trade costs), d_{·,EG,s} = 1.0 (no domestic shipping cost)."),
        sp(6),

        h2("4.2  MFN tariff for US exports"),
        p("Non-compliant firms exporting to the US face a <b>Most Favored Nation (MFN) "
          "tariff</b>  t^MFN_s  on top of the iceberg cost. The total delivered wedge is:"),
        eq("τ_{rjs}  =  d_{rjs} × { 1 + t^MFN_s   if j=US and non-compliant"),
        eq("                       { 1              if j=US and compliant (QIZ)"),
        eq("                       { 1              if j ∈ {EG, RW}"),
        p("Compliant QIZ firms pay <b>zero tariff</b> — that is the core benefit of "
          "the QIZ arrangement. In the baseline: t^MFN_T = 0.15, t^MFN_O = 0.03."),
        sp(6),

        h2("4.3  Delivered price"),
        p("The firm sets price at markup over delivered marginal cost:"),
        eq("p_{rjs}(φ)  =  μ_s · τ_{rjs} · mc_{rs}(φ)"),
        p("In the code:"),
        code(
            "p_EG = mu * (tau_EG * mc_base)\n"
            "p_RW = mu * (tau_RW * mc_base)\n"
            "p_US = mu * (tau_US * mc_US)   # mc_US differs for compliant firms"
        ),
        p("Note: <i>mc_US</i> can differ from <i>mc_base</i> for compliant firms, "
          "because compliance changes the intermediate input used for US-destined "
          "production (see Section 6)."),
        PageBreak(),
    ]

    # ── Section 5: Export Participation ──────────────────────────────────────
    story += [
        h1("5.  Export Participation"),
        rule(),
        h2("5.1  The fixed cost condition"),
        p("Serving each destination requires paying a <b>fixed cost</b>  w_r · f_{rjs}  "
          "(in units of labor). A firm serves destination j if and only if variable "
          "profit covers the fixed cost:"),
        eq("R_{rjs}(φ) / σ_s  ≥  w_r · f_{rjs}"),
        p("Since  R_{rjs} ∝ φ^(σ_s−1)  (more productive firms charge lower prices "
          "and capture higher revenue shares), there exists a <b>productivity cutoff</b> "
          "φ*_{rjs} such that only firms with φ ≥ φ*_{rjs} find it profitable to "
          "serve market j. In the code:"),
        code(
            "serve_EG = (pi_EG >= w_r * f_dom)\n"
            "serve_US = (pi_US >= w_r * f_US)\n"
            "serve_RW = (pi_RW >= w_r * f_RW)"
        ),
        sp(6),

        h2("5.2  Market hierarchy"),
        p("In practice, the domestic market (EG) is the easiest to serve: it has the "
          "lowest fixed cost and no iceberg losses. Exporting to US or RW requires "
          "additional fixed costs and trade costs. This generates a natural hierarchy:"),
        bullet("All exporters also serve the domestic market."),
        bullet("Not all domestic sellers export."),
        bullet("US exporters face a higher fixed cost than RW exporters (in the baseline)."),
        sp(6),

        h2("5.3  Pareto productivity distribution"),
        p("Each entrant in (r, s) draws productivity φ from a Pareto distribution:"),
        eq("G_s(φ)  =  1 − (φ_min,s / φ)^(θ_s),        φ ≥ φ_min,s"),
        p("The shape parameter θ_s must satisfy  θ_s > σ_s − 1  to ensure finite "
          "expected firm size. Higher θ_s concentrates more firms near the minimum — "
          "a fatter left tail of low-productivity firms."),
        p("In the code, the Pareto distribution is approximated numerically using "
          "a <b>midpoint quadrature rule</b>:"),
        code(
            "def pareto_grid(phi_min, theta, n):\n"
            "    u = (np.arange(n) + 0.5) / n      # uniform draws in (0,1)\n"
            "    phi = phi_min * (1.0-u)**(-1.0/theta)  # inverse Pareto CDF\n"
            "    w = np.full(n, 1.0/n)              # equal probability weights\n"
            "    return phi, w"
        ),
        p("The <i>u</i> values are evenly spaced interior points; the inverse CDF "
          "maps them to the Pareto support; weights are uniform because the grid "
          "is uniform over probability mass."),
        PageBreak(),
    ]

    # ── Section 6: ROO Compliance ─────────────────────────────────────────────
    story += [
        h1("6.  Rules of Origin (ROO) Compliance"),
        rule(),
        h2("6.1  What is the ROO?"),
        p("The QIZ agreement grants tariff-free access to the US market on the "
          "condition that the exported product contains at least a fraction <i>γ_s</i> "
          "of Israeli inputs. This creates a <b>trade-off</b>:"),
        bullet("<b>Benefit:</b> zero US tariff instead of t^MFN_s."),
        bullet("<b>Cost:</b> must use more expensive Israeli inputs, plus administrative burden."),
        p("Only firms in the <b>QIZ-eligible region</b> (r = Q) can comply. "
          "Non-QIZ firms (r = N) cannot access the preferential rate regardless."),
        sp(6),

        h2("6.2  The intermediate input cost under compliance"),
        p("A compliant firm must use an input bundle with fraction γ_s Israeli content. "
          "This is modeled as a <b>normalized Cobb-Douglas</b> quantity index:"),
        eq("m  =  m_IL^(γ_s) · m_RW^(1−γ_s)  /  [ γ_s^(γ_s) · (1−γ_s)^(1−γ_s) ]"),
        p("The denominator  K = γ_s^(γ_s) · (1−γ_s)^(1−γ_s)  is the <b>normalization "
          "constant</b>. This is essential."),
        sp(6),

        h2("6.3  Why normalization matters"),
        p("Without normalization, switching from free-market sourcing to any Cobb-Douglas "
          "mix would mechanically inflate unit cost — even if Israeli and world inputs "
          "cost exactly the same. The normalization ensures that if  p_IL = p_RW,  "
          "compliance has <b>zero marginal cost in unit cost terms</b>."),
        p("The ROO only bites when Israeli inputs are genuinely more expensive than world inputs."),
        sp(6),

        h2("6.4  Deriving the compliant unit cost"),
        p("The unit cost of the normalized input bundle is found by cost minimization "
          "over (m_IL, m_RW). The normalization constant in the quantity index cancels "
          "in the derivation, leaving:"),
        eq("c^mix_{m,s}(γ_s)  =  p_IL,s^(γ_s) · p_RW,s^(1−γ_s)"),
        p("This is a weighted geometric mean of input prices."),
        p("<b>Verification:</b> if  p_IL = p_RW = p,  then  c^mix = p^(γ_s) · p^(1−γ_s) = p. "
          "Cost equals the common input price — no compliance penalty. ✓"),
        p("In the code:"),
        code(
            "def cmix_normalized(p_il, p_rw, gamma):\n"
            "    return (p_il ** gamma) * (p_rw ** (1.0 - gamma))"
        ),
        sp(6),

        h2("6.5  Administrative wedge"),
        p("Beyond input costs, compliance creates paperwork and coordination costs "
          "that increase with the stringency γ_s:"),
        eq("χ_s(γ_s)  =  1 + ξ_s · γ_s,          ξ_s > 0"),
        p("The total compliant intermediate cost for US-destined shipments is:"),
        eq("c^{C,US}_{m,s}  =  χ_s(γ_s) · c^mix_{m,s}(γ_s)"),
        p("And the compliant marginal cost for US production:"),
        eq("mc^C_US  =  (1/φ) · w_r^(α_s) · (c^{C,US}_{m,s})^(1−α_s)  /  [α_s^α_s · (1−α_s)^(1−α_s)]"),
        p("In the baseline: ξ_T = 1.0, ξ_O = 0.6, γ_T = γ_O = 0.105."),
        sp(6),

        h2("6.6  ROO applies only to US shipments"),
        p("This is a deliberate and important modeling choice. Non-US production "
          "(domestic EG, RW exports) uses the free-market input  c^N_m = p_RW,s. "
          "Only US-destined output must satisfy the Israeli content requirement. "
          "This is consistent with how QIZ certification works in practice: "
          "certification is at the <b>shipment level</b>, not the firm level."),
        sp(6),

        h2("6.7  Partial compliance — idiosyncratic fixed costs"),
        p("Even among firms that benefit from compliance in expectation, not all comply. "
          "Each firm draws an idiosyncratic compliance fixed cost:"),
        eq("f^C_{i,s}  =  f^C_s · exp(ε_i),          ε_i ~ N(0, σ^2_{C,s})"),
        p("So  f^C_{i,s}  is <b>lognormally distributed</b>: the median compliance "
          "cost is  f^C_s,  but some firms face much higher costs (e.g., their supply "
          "chain is hard to restructure). This generates <b>partial take-up</b>: some "
          "firms comply, some don't, even among those serving the US."),
        note("f^C_s  (fC_mean in the code) is the <b>median</b>, not the mean, of "
             "the lognormal. The true mean is  fC_mean · exp(σ_C² / 2). For textiles: "
             "mean ≈ 0.35 · exp(0.125) ≈ 0.397. This matters if calibrating to survey "
             "data on average compliance expenditure."),
        sp(6),

        h2("6.8  The compliance decision in the code"),
        p("For Q-region firms, the code enumerates <b>four strategy combinations</b>:"),
        build_table(
            ["comply", "upgrade", "Description"],
            [
                ["False", "False", "Baseline: no compliance, no upgrade"],
                ["False", "True",  "Upgrade only (if US is served)"],
                ["True",  "False", "Comply with ROO, no upgrade"],
                ["True",  "True",  "Comply and upgrade"],
            ],
            col_widths=[2.5*cm, 2.5*cm, 9*cm],
        ),
        sp(4),
        p("For each strategy, the code computes total profit and picks the best. "
          "Compliance is only effective if the firm actually serves the US — "
          "compliance without US access is a dominated strategy and is skipped."),
        PageBreak(),
    ]

    # ── Section 7: Productivity Upgrading ────────────────────────────────────
    story += [
        h1("7.  Productivity Upgrading"),
        rule(),
        h2("7.1  The mechanism"),
        p("Serving the US market exposes firms to demanding buyers, quality standards, "
          "and global value chains — this triggers <b>productivity upgrading</b>. "
          "Any firm that serves the US can pay a fixed cost  w_r · f^U_s  to "
          "permanently scale its productivity:"),
        eq("φ'  =  δ_s · φ,          δ_s > 1"),
        p("In the baseline: δ_T = 1.12, δ_O = 1.07, f^U_T = 1.8, f^U_O = 2.3 (labor units)."),
        sp(6),

        h2("7.2  Why upgrading helps across all markets"),
        p("Under CES demand with markup pricing, revenue in any destination j satisfies:"),
        eq("R_{rjs}  ∝  φ^(σ_s − 1)"),
        p("This is because lower mc ∝ 1/φ translates to a lower price, which "
          "expands demand. After upgrading, φ' = δ_s · φ, so:"),
        eq("R'_{rjs} / R_{rjs}  =  δ_s^(σ_s − 1)"),
        p("This ratio is <b>identical across all destinations</b>. Upgrading lifts "
          "revenues in Egypt, the US, <i>and</i> the Rest of World simultaneously. "
          "This is the key <b>spillover mechanism</b>: firms that upgrade because of "
          "US access also become more competitive in third markets."),
        sp(6),

        h2("7.3  The US-access constraint"),
        p("Upgrading is only available if the firm serves the US "
          "(parameter <i>upgrade_requires_US = True</i>). In the code:"),
        code(
            "if upgrade and p['upgrade_requires_US'] and (not serve_US):\n"
            "    eff_upgrade = False   # cannot upgrade without US market access"
        ),
        p("If a firm plans to upgrade but does not end up serving the US "
          "(because the profit threshold is not met), the upgrade is dropped and "
          "profits are recomputed without it. This reflects the economic story: "
          "upgrading is triggered by learning-by-exporting to the US."),
        PageBreak(),
    ]

    # ── Section 8: General Equilibrium ───────────────────────────────────────
    story += [
        h1("8.  General Equilibrium"),
        rule(),
        h2("8.1  Labor mobility"),
        p("Workers choose between regions Q and N based on real wages. "
          "The <b>logit</b> allocation is:"),
        eq("λ_r  =  (w_r / P_EG)^κ  /  Σ_{r'} (w_{r'} / P_EG)^κ"),
        eq("L_r  =  λ_r · L"),
        p("Higher κ means workers are more responsive to wage differences "
          "(more mobile). At κ → ∞ wages equalize. At κ = 0 labor is immobile. "
          "Baseline: κ = 2.0."),
        sp(6),

        h2("8.2  Free entry"),
        p("In each (r, s), firms pay entry cost  w_r · f^E_{rs}  to draw a productivity. "
          "In equilibrium, the <b>expected profit from entry equals the entry cost</b>:"),
        eq("E[Π_{rs}(φ, ε)]  =  w_r · f^E_{rs}"),
        p("This pins down the mass of entrants M_{rs}. If expected profits exceed "
          "entry cost, more firms enter, driving down profits (via the price index) "
          "until equality holds."),
        sp(6),

        h2("8.3  Labor market clearing"),
        p("In each region, total labor demand must equal labor supply:"),
        eq("L_r  =  Σ_s  M_{rs} · [ E[l^var_{rs}]  +  E[l^fix_{rs}]  +  f^E_{rs} ]"),
        p("where:"),
        bullet("E[l^var_{rs}]  = expected variable labor from production"),
        bullet("E[l^fix_{rs}]  = expected fixed labor (domestic, export, compliance, upgrade costs)"),
        bullet("f^E_{rs}       = entry cost per entrant"),
        sp(6),

        h2("8.4  The outer loop solver"),
        p("The solver updates wages and entry masses using a log-linear tatonnement:"),
        code(
            "# If labor demand > supply: raise wage\n"
            "w[r]   *= exp(step * log(Ld[r] / Ls[r]))\n"
            "\n"
            "# If E[profit] > w*f_E: too profitable, more firms enter\n"
            "M[r,s] *= exp(step * log(E_profit / entry_cost))"
        ),
        p("This adjusts wages up when labor demand exceeds supply, and entry masses "
          "up when profits exceed entry costs. The step size  outer_step = 0.10  "
          "controls the speed (smaller = more stable, slower)."),
        sp(6),

        h2("8.5  Welfare"),
        p("Welfare is real income — how much the composite good Egyptian households "
          "can afford:"),
        eq("W  =  Y_EG / P_EG"),
        PageBreak(),
    ]

    # ── Section 9: Counterfactuals ────────────────────────────────────────────
    story += [
        h1("9.  Counterfactuals"),
        rule(),
        h2("9.1  Counterfactual A — Shut down the productivity channel"),
        p("Set  δ_s = 1  for all sectors (no upgrading benefit). Re-solve the full "
          "general equilibrium. Compare with baseline across:"),
        bullet("<b>Compliance rates:</b> without upgrading, the only benefit of "
               "compliance is tariff-free US access. Compliance rates may fall."),
        bullet("<b>RW exports:</b> without the upgrading spillover, compliant firms "
               "won't gain RW competitiveness. This tests whether upgrading drives "
               "the observed pattern."),
        bullet("<b>Wages and welfare:</b> if upgrading is important, shutting it down "
               "reduces wages in Q and aggregate welfare."),
        p("In the code:"),
        code("p_off['delta'] = {s: 1.0 for s in p['sectors']}"),
        sp(8),

        h2("9.2  Counterfactual B — Vary the ROO content requirement γ_T"),
        p("Solve GE for a grid of  γ_T ∈ [0, γ̄_T],  holding everything else fixed. "
          "For each γ_T, record: welfare, wages, compliance rates, US exports, "
          "RW exports, and implied Israeli input intensity."),
        p("<b>What to expect along the γ_T path:</b>"),
        bullet("<b>γ_T = 0:</b> no Israeli content required. Compliance is cheap "
               "(no input cost change, only paperwork). High compliance rate."),
        bullet("<b>γ_T increases:</b> compliance becomes more costly. Fewer firms "
               "comply. But those that do use more Israeli content."),
        bullet("<b>Welfare W(γ_T):</b> traces the policy frontier — the trade-off "
               "between Israeli conditionality and Egyptian welfare gains from "
               "tariff-free US access."),
        p("The baseline uses  γ_T = γ_O = 0.105  (10.5% Israeli content requirement)."),
        PageBreak(),
    ]

    # ── Section 10: Bugs & Fixes ──────────────────────────────────────────────
    story += [
        h1("10.  Bugs Found and Corrections Applied"),
        rule(),
        p("The following issues were identified in <i>qiz_model_ge.py</i>. "
          "All have been corrected in the source file."),
        sp(8),

        h2("Bug 1 — serve_EG not rechecked after upgrade fallback  [HIGH]"),
        Paragraph(
            "<b>Location:</b> firm_best(), lines ~286–321<br/><br/>"
            "<b>What happened:</b> When a firm plans to upgrade (upgrade=True) but fails "
            "to serve the US — so the upgrade is dropped — the code recomputes R_EG under "
            "the non-upgrade marginal cost. However, the flag serve_EG was never updated. "
            "A firm only marginally active in Egypt <i>because</i> of the upgrade boost "
            "was incorrectly recorded as active after the upgrade was dropped.<br/><br/>"
            "<b>Economic consequence:</b> Corrupts the domestic price index "
            "(a non-viable variety enters it), overstates labor demand in Egypt, "
            "and inflates free-entry expected profits.<br/><br/>"
            "<b>Fix applied:</b> After the upgrade fallback block, serve_EG is now "
            "rechecked: if R_EG/sigma &lt; w_r*f_dom after losing the upgrade, "
            "the firm is set to fully inactive.",
            BUG_HIGH
        ),
        sp(8),

        h2("Bug 2 — normal_grid uses endpoint-inclusive linspace  [MEDIUM]"),
        Paragraph(
            "<b>Location:</b> normal_grid(), lines ~154–162<br/><br/>"
            "<b>What happened:</b> The docstring claims 'midpoint-rule quadrature' "
            "but np.linspace(-a, a, n) includes the endpoints −a and +a. "
            "A true midpoint rule uses only interior points. The step size "
            "h = 2a/(n−1) was also the linspace spacing, not the midpoint spacing 2a/n.<br/><br/>"
            "<b>Fix applied:</b> Replaced linspace with the midpoint formula:<br/>"
            "x = −a + (arange(n) + 0.5) × (2a/n),   h = 2a/n",
            BUG_MED
        ),
        sp(8),

        h2("Bug 3 — print_key references p which is not in scope  [HIGH]"),
        Paragraph(
            "<b>Location:</b> print_key(), line ~637<br/><br/>"
            "<b>What happened:</b> The function print_key(sol, label) used p['L_total'] "
            "but p was not a parameter. It worked only because p was defined globally "
            "in __main__. If called from any other context, it raises NameError.<br/><br/>"
            "<b>Fix applied:</b> Added p as an explicit third parameter: "
            "print_key(sol, label, p).",
            BUG_HIGH
        ),
        sp(8),

        h2("Bug 4 — A line in print_key always printed 1.0  [MEDIUM]"),
        Paragraph(
            "<b>Location:</b> print_key(), line ~636<br/><br/>"
            "<b>What happened:</b> The expression "
            "sol['Ls'][r] / sol['goods']['Ls'][r] always equals 1.0 because "
            "sol['Ls'] and sol['goods']['Ls'] are the <i>same Python object</i> "
            "(both reference the same dict). The comment even admitted 'placeholder'. "
            "This printed {'Q': 1.0, 'N': 1.0} every time, providing no information.<br/><br/>"
            "<b>Fix applied:</b> Removed the redundant line entirely.",
            BUG_MED
        ),
        sp(8),

        h2("Bug 5 — Compliance fallback used −1e18 sentinel  [MEDIUM]"),
        Paragraph(
            "<b>Location:</b> firm_best(), lines ~328–337<br/><br/>"
            "<b>What happened:</b> When comply=True but serve_US=False, the code set "
            "profit = −1e18 to ensure this dominated strategy never wins. This is "
            "fragile: it relies on (comply=False, upgrade=False) always being in "
            "the strategy set. Any future refactoring that reorders or removes strategies "
            "could silently break this.<br/><br/>"
            "<b>Fix applied:</b> Replaced the sentinel with a clean 'continue' to "
            "skip the dominated strategy entirely.",
            BUG_MED
        ),
        sp(8),

        h2("Bug 6 — Shallow copy in counterfactual  [LOW]"),
        Paragraph(
            "<b>Location:</b> counterfactual_shutdown_productivity(), line ~600<br/><br/>"
            "<b>What happened:</b> p_off = {k: v for k,v in p.items()} is a shallow copy. "
            "It is currently safe because p_off['delta'] is immediately reassigned to a "
            "new dict. But in-place mutation of any nested dict would silently corrupt "
            "the original p.<br/><br/>"
            "<b>Fix applied:</b> Replaced with copy.deepcopy(p).",
            BUG_LOW
        ),
        sp(8),

        h2("Calibration Note — fC_mean is the median, not the mean"),
        Paragraph(
            "<b>Location:</b> params_defensible(), lines ~109–110<br/><br/>"
            "The compliance cost fC_i = fC_mean × exp(σ_C × ε) defines a lognormal "
            "where fC_mean is the <b>median</b>. The <b>mean</b> is "
            "fC_mean × exp(σ_C² / 2). For textiles: mean ≈ 0.35 × exp(0.125) ≈ 0.397 "
            "(13% above the median). If calibrating to average compliance expenditure "
            "from survey data, the calibration target should be the mean, not fC_mean.",
            BUG_LOW
        ),
        sp(8),

        h2("Modeling Note — No explicit wage numeraire"),
        Paragraph(
            "<b>Location:</b> solve_equilibrium(), line ~521<br/><br/>"
            "No wage is held fixed during iteration. The model is implicitly pinned "
            "by the exogenous foreign price indices P_foreign = 1.0. If foreign prices "
            "are ever changed, the absolute wage level becomes unanchored. "
            "Best practice: after each update step, normalize one wage "
            "(e.g., w_Q = 1.0) and rescale accordingly.",
            BUG_LOW
        ),
        sp(12),
        rule(),
        sp(6),
        p("<i>All six bugs have been corrected in the source file qiz_model_ge.py. "
          "The two notes are flagged for calibration awareness but do not require "
          "code changes for the current baseline.</i>"),
    ]

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"PDF saved to: {OUTPUT}")


if __name__ == "__main__":
    build()
