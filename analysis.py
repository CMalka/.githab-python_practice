"""
Full retail dataset analysis: categories, customers, order_details, orders, products.
Produces a KPI summary tile and 8 charts covering trend, category, product,
customer, geography, returns, and timing patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# 0. Style (validated palette — see dataviz skill references/palette.md)
# ---------------------------------------------------------------------------
SURFACE       = "#fcfcfb"
PAGE          = "#f9f9f7"
INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"
GRID          = "#e1e0d9"
BASELINE      = "#c3c2b7"

CAT = {
    "blue":    "#2a78d6",
    "orange":  "#eb6834",
    "aqua":    "#1baf7a",
    "yellow":  "#eda100",
    "magenta": "#e87ba4",
    "green":   "#008300",
    "violet":  "#4a3aa7",
    "red":     "#e34948",
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS_CRITICAL = "#d03b3b"
STATUS_GOOD = "#0ca30c"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.grid": False,
    "font.size": 11,
})

def style_axes(ax, y_grid=True, x_grid=False):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(1)
    if y_grid:
        ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    if x_grid:
        ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

def money(x, pos=None):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"

MONEY_FMT = mticker.FuncFormatter(money)

# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
categories = pd.read_csv("categories.csv")
products   = pd.read_csv("products.csv")
customers  = pd.read_csv("customers.csv")
orders     = pd.read_csv("orders.csv")
details    = pd.read_csv("order_details.csv")

orders["OrderDate"] = pd.to_datetime(orders["OrderDate"])
customers["SignUpDate"] = pd.to_datetime(customers["SignUpDate"])

# Sentinel 9999-12-31 in ReturnDate marks "never returned" -> keep IsReturned as source of truth
details["IsReturned"] = details["IsReturned"].astype(int)
details["ReturnDate"] = pd.to_datetime(details["ReturnDate"], errors="coerce")
details.loc[details["ReturnDate"].dt.year == 9999, "ReturnDate"] = pd.NaT

details["Revenue"] = details["Quantity"] * details["UnitPrice"] * (1 - details["DiscountRate"])
details["Cost"]    = details["Quantity"] * details["UnitCost"]
details["Profit"]  = details["Revenue"] - details["Cost"]

# Master denormalized table
df = (details
      .merge(orders, on="OrderID", how="left")
      .merge(products, on="ProductID", how="left")
      .merge(categories, on="CategoryID", how="left")
      .merge(customers, on="CustomerID", how="left"))

df["OrderMonth"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()
df["DayOfWeek"]  = df["OrderDate"].dt.day_name()
df["OrderHour"]  = pd.to_datetime(df["OrderTime"], format="%H:%M:%S", errors="coerce").dt.hour

print(f"Loaded {len(orders):,} orders / {len(details):,} line items / "
      f"{len(customers):,} customers / {len(products):,} products / {len(categories)} categories")

# ---------------------------------------------------------------------------
# 2. KPIs
# ---------------------------------------------------------------------------
total_revenue   = df["Revenue"].sum()
total_profit    = df["Profit"].sum()
total_orders    = orders["OrderID"].nunique()
total_customers = customers["CustomerID"].nunique()
aov             = total_revenue / total_orders
return_rate     = details["IsReturned"].mean()
avg_discount    = details.loc[details["DiscountRate"] > 0, "DiscountRate"].mean()

kpis = [
    ("Total Revenue",   money(total_revenue, None)),
    ("Total Profit",    money(total_profit, None)),
    ("Total Orders",    f"{total_orders:,}"),
    ("Total Customers", f"{total_customers:,}"),
    ("Avg Order Value", money(aov, None)),
    ("Return Rate",     f"{return_rate*100:.1f}%"),
]

fig, ax = plt.subplots(figsize=(13, 2.6))
fig.patch.set_facecolor(PAGE)
ax.set_facecolor(PAGE)
ax.axis("off")
n = len(kpis)
w = 1 / n
for i, (label, value) in enumerate(kpis):
    x0 = i * w
    box = FancyBboxPatch((x0 + 0.006, 0.06), w - 0.012, 0.88,
                          boxstyle="round,pad=0,rounding_size=0.03",
                          transform=ax.transAxes, facecolor=SURFACE,
                          edgecolor=GRID, linewidth=1)
    ax.add_patch(box)
    ax.text(x0 + w / 2, 0.62, value, transform=ax.transAxes,
            ha="center", va="center", fontsize=19, fontweight="bold", color=INK_PRIMARY)
    ax.text(x0 + w / 2, 0.24, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=10.5, color=INK_SECONDARY)
fig.suptitle("Retail Performance — Key Metrics", x=0.02, ha="left",
             fontsize=14, fontweight="bold", color=INK_PRIMARY, y=1.02)
plt.tight_layout()
plt.savefig("chart_0_kpi_summary.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 3. Revenue & order trend (small multiples, shared x, NOT dual-axis)
# ---------------------------------------------------------------------------
monthly = df.groupby("OrderMonth").agg(Revenue=("Revenue", "sum"),
                                        Orders=("OrderID", "nunique")).reset_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.plot(monthly["OrderMonth"], monthly["Revenue"], color=CAT["blue"], linewidth=2)
ax1.fill_between(monthly["OrderMonth"], monthly["Revenue"], color=CAT["blue"], alpha=0.08)
ax1.yaxis.set_major_formatter(MONEY_FMT)
ax1.set_title("Monthly Revenue", loc="left", fontsize=12, fontweight="bold", color=INK_PRIMARY)
style_axes(ax1)

ax2.plot(monthly["OrderMonth"], monthly["Orders"], color=CAT["orange"], linewidth=2)
ax2.fill_between(monthly["OrderMonth"], monthly["Orders"], color=CAT["orange"], alpha=0.08)
ax2.set_title("Monthly Order Count", loc="left", fontsize=12, fontweight="bold", color=INK_PRIMARY)
style_axes(ax2)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("chart_1_monthly_trend.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 4. Revenue by category
# ---------------------------------------------------------------------------
cat_rev = df.groupby("CategoryName")["Revenue"].sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(11, 6))
colors = [SEQ_BLUE[3]] * len(cat_rev)
colors[-1] = CAT["blue"]  # top category emphasized
bars = ax.barh(cat_rev.index, cat_rev.values, color=colors, height=0.62, zorder=3)
for b, v in zip(bars, cat_rev.values):
    ax.text(v + cat_rev.max()*0.01, b.get_y() + b.get_height()/2, money(v, None),
            va="center", fontsize=10, color=INK_SECONDARY)
ax.xaxis.set_major_formatter(MONEY_FMT)
ax.set_title("Revenue by Category", loc="left", fontsize=13, fontweight="bold", color=INK_PRIMARY)
style_axes(ax, y_grid=False, x_grid=True)
plt.tight_layout()
plt.savefig("chart_2_revenue_by_category.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 5. Top 15 products by revenue
# ---------------------------------------------------------------------------
prod_rev = df.groupby("ProductName")["Revenue"].sum().sort_values(ascending=False).head(15).sort_values()
fig, ax = plt.subplots(figsize=(11, 7.5))
bars = ax.barh(prod_rev.index, prod_rev.values, color=SEQ_BLUE[3], height=0.62, zorder=3)
bars[-1].set_color(CAT["blue"])
for b, v in zip(bars, prod_rev.values):
    ax.text(v + prod_rev.max()*0.01, b.get_y() + b.get_height()/2, money(v, None),
            va="center", fontsize=9.5, color=INK_SECONDARY)
ax.xaxis.set_major_formatter(MONEY_FMT)
ax.set_title("Top 15 Products by Revenue", loc="left", fontsize=13, fontweight="bold", color=INK_PRIMARY)
style_axes(ax, y_grid=False, x_grid=True)
plt.tight_layout()
plt.savefig("chart_3_top_products.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 6. Customer segment & region
# ---------------------------------------------------------------------------
seg_rev = df.groupby("CustomerSegment")["Revenue"].sum().sort_values(ascending=False)
reg_rev = df.groupby("Region")["Revenue"].sum().sort_values(ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
seg_colors = [CAT["blue"], CAT["orange"], CAT["aqua"], CAT["yellow"]][:len(seg_rev)]
bars1 = ax1.bar(seg_rev.index, seg_rev.values, color=seg_colors, width=0.55, zorder=3)
for b, v in zip(bars1, seg_rev.values):
    ax1.text(b.get_x()+b.get_width()/2, v + seg_rev.max()*0.015, money(v, None),
              ha="center", fontsize=10, color=INK_SECONDARY)
ax1.yaxis.set_major_formatter(MONEY_FMT)
ax1.set_title("Revenue by Customer Segment", loc="left", fontsize=12, fontweight="bold", color=INK_PRIMARY)
style_axes(ax1)

bars2 = ax2.barh(reg_rev.index, reg_rev.values, color=SEQ_BLUE[3], height=0.6, zorder=3)
bars2[-1].set_color(CAT["blue"])
for b, v in zip(bars2, reg_rev.values):
    ax2.text(v + reg_rev.max()*0.01, b.get_y()+b.get_height()/2, money(v, None),
              va="center", fontsize=9.5, color=INK_SECONDARY)
ax2.xaxis.set_major_formatter(MONEY_FMT)
ax2.set_title("Revenue by Region", loc="left", fontsize=12, fontweight="bold", color=INK_PRIMARY)
style_axes(ax2, y_grid=False, x_grid=True)
plt.tight_layout()
plt.savefig("chart_4_segment_region.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 7. Returns analysis: rate by category + top reasons
# ---------------------------------------------------------------------------
ret_by_cat = df.groupby("CategoryName")["IsReturned"].mean().sort_values(ascending=True) * 100
overall_rate = return_rate * 100

reasons = (df.loc[df["IsReturned"] == 1, "ReturnReason"]
             .dropna()
             .loc[lambda s: s.str.lower() != "none"]
             .value_counts()
             .head(8)
             .sort_values())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
bar_colors = [STATUS_CRITICAL if v > overall_rate else SEQ_BLUE[3] for v in ret_by_cat.values]
bars = ax1.barh(ret_by_cat.index, ret_by_cat.values, color=bar_colors, height=0.62, zorder=3)
ax1.axvline(overall_rate, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=2)
ax1.text(overall_rate, len(ret_by_cat)-0.3, f" avg {overall_rate:.1f}%", color=INK_MUTED,
         fontsize=9, va="bottom")
for b, v in zip(bars, ret_by_cat.values):
    ax1.text(v + ret_by_cat.max()*0.015, b.get_y()+b.get_height()/2, f"{v:.1f}%",
              va="center", fontsize=9.5, color=INK_SECONDARY)
ax1.set_title("Return Rate by Category", loc="left", fontsize=12, fontweight="bold", color=INK_PRIMARY)
ax1.set_xlabel("% of line items returned")
style_axes(ax1, y_grid=False, x_grid=True)

if len(reasons) > 0:
    bars2 = ax2.barh(reasons.index, reasons.values, color=CAT["orange"], height=0.6, zorder=3)
    for b, v in zip(bars2, reasons.values):
        ax2.text(v + reasons.max()*0.015, b.get_y()+b.get_height()/2, f"{v:,}",
                  va="center", fontsize=9.5, color=INK_SECONDARY)
ax2.set_title("Top Return Reasons", loc="left", fontsize=12, fontweight="bold", color=INK_PRIMARY)
style_axes(ax2, y_grid=False, x_grid=True)
plt.tight_layout()
plt.savefig("chart_5_returns.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 8. Order timing heatmap: day-of-week x hour
# ---------------------------------------------------------------------------
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heat = (df.drop_duplicates("OrderID")
          .groupby(["DayOfWeek", "OrderHour"])
          .size()
          .unstack(fill_value=0)
          .reindex(dow_order))
heat = heat.reindex(columns=range(24), fill_value=0)

fig, ax = plt.subplots(figsize=(13, 5))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
im = ax.imshow(heat.values, aspect="auto", cmap=cmap)
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
ax.set_yticks(range(len(dow_order)))
ax.set_yticklabels(dow_order, fontsize=10)
ax.set_title("Order Volume by Day of Week & Hour", loc="left", fontsize=13, fontweight="bold", color=INK_PRIMARY)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
cbar.set_label("Orders", color=INK_SECONDARY, fontsize=9)
cbar.ax.tick_params(labelsize=8, colors=INK_MUTED)
plt.tight_layout()
plt.savefig("chart_6_order_heatmap.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 9. Customer demographics: age distribution + revenue by gender
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
ax1.hist(customers["Age"], bins=20, color=SEQ_BLUE[3], edgecolor=SURFACE, linewidth=0.8, zorder=3)
ax1.axvline(customers["Age"].mean(), color=CAT["orange"], linestyle="--", linewidth=1.5, zorder=4)
ax1.text(customers["Age"].mean(), ax1.get_ylim()[1]*0.95,
          f" avg {customers['Age'].mean():.0f}", color=CAT["orange"], fontsize=9, va="top")
ax1.set_title("Customer Age Distribution", loc="left", fontsize=12, fontweight="bold", color=INK_PRIMARY)
ax1.set_xlabel("Age")
style_axes(ax1)

gender_rev = df.groupby("Gender")["Revenue"].sum().sort_values(ascending=False)
g_colors = [CAT["blue"], CAT["orange"], CAT["aqua"]][:len(gender_rev)]
bars = ax2.bar(gender_rev.index, gender_rev.values, color=g_colors, width=0.5, zorder=3)
for b, v in zip(bars, gender_rev.values):
    ax2.text(b.get_x()+b.get_width()/2, v + gender_rev.max()*0.015, money(v, None),
              ha="center", fontsize=10, color=INK_SECONDARY)
ax2.yaxis.set_major_formatter(MONEY_FMT)
ax2.set_title("Revenue by Gender", loc="left", fontsize=12, fontweight="bold", color=INK_PRIMARY)
style_axes(ax2)
plt.tight_layout()
plt.savefig("chart_7_demographics.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 10. Text summary
# ---------------------------------------------------------------------------
top_cat = cat_rev.idxmax()
top_prod = prod_rev.idxmax()
top_region = reg_rev.idxmax()
worst_return_cat = ret_by_cat.idxmax()

summary = f"""
RETAIL DATA ANALYSIS — SUMMARY
================================
Period: {df['OrderDate'].min().date()} to {df['OrderDate'].max().date()}

Headline numbers:
  Total revenue:    {money(total_revenue, None)}
  Total profit:     {money(total_profit, None)}  ({total_profit/total_revenue*100:.1f}% margin)
  Total orders:     {total_orders:,}
  Total customers:  {total_customers:,}
  Avg order value:  {money(aov, None)}
  Return rate:      {return_rate*100:.1f}% of line items
  Avg discount (on discounted items): {avg_discount*100:.1f}%

Top performers:
  #1 category:  {top_cat} ({money(cat_rev.max(), None)})
  #1 product:   {top_prod} ({money(prod_rev.max(), None)})
  #1 region:    {top_region} ({money(reg_rev.max(), None)})

Watch list:
  Highest return-rate category: {worst_return_cat} ({ret_by_cat.max():.1f}%, vs {overall_rate:.1f}% avg)
"""
print(summary)
with open("summary.txt", "w") as f:
    f.write(summary)

print("Done. Generated chart_0..chart_7 PNGs + summary.txt")
