import sys
sys.path.append(r"C:\Users\hdagne1\Box\NRT_Project_2025Fall\Habtamu\HydroAuditToolFrameowrk")

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Scripts.metrics import calc_nse
from matplotlib.dates import DateFormatter

# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"C:\Users\hdagne1\Box\NRT_Project_2025Fall\Habtamu\HydroAuditToolFrameowrk")
RUN_DIR = BASE_DIR  / "runs"     # FIXED PATH
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# CAMELS validation period
VAL_START = pd.to_datetime('01101995', format='%d%m%Y')
VAL_END = pd.to_datetime('30092010', format='%d%m%Y')
ONE_YEAR_END = VAL_START + pd.DateOffset(years=1)

# =========================
# Find a .p result file
# =========================
run_dirs = sorted(RUN_DIR.glob("run_*"))
if not run_dirs:
    raise FileNotFoundError("❌ No run_* folders found under src/runs/")

result_file = None
for rd in run_dirs:
    files = list(rd.glob("lstm_no_static*.p"))
    if files:
        result_file = files[0]
        break

if result_file is None:
    raise FileNotFoundError("❌ No .p file found inside run_* folders!")

print(f"📌 Loaded LSTM run: {result_file}")

with open(result_file, "rb") as fp:
    results = pickle.load(fp)

# =========================
# KGE function
# =========================
def calc_kge(obs, sim):
    obs = obs.flatten()
    sim = sim.flatten()
    r = np.corrcoef(obs, sim)[0, 1]
    beta = np.mean(sim) / np.mean(obs)
    alpha = np.std(sim) / np.std(obs)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

# =========================
# Loop basins → metrics + plots
# =========================
metrics = []

for basin_id, df in results.items():

    df = df.copy()

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    # Clip to validation period
    df = df.loc[(df.index >= VAL_START) & (df.index <= VAL_END)]
    if df.empty:
        print(f"⚠ Skipping {basin_id}: No data in validation period.")
        continue

    # Use only the first 1-year of validation
    df_plot = df.loc[(df.index >= VAL_START) & (df.index < ONE_YEAR_END)]
    if df_plot.empty:
        print(f"⚠ Skipping {basin_id}: No data in first validation year.")
        continue

    qobs = df_plot["qobs"].values
    qsim = df_plot["qsim"].values

    # Compute metrics on the full validation period
    nse = calc_nse(df["qobs"].values, df["qsim"].values)
    kge = calc_kge(df["qobs"].values, df["qsim"].values)

    metrics.append([basin_id, nse, kge])

    # =========================
    # Plot time series
    # =========================
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_plot.index, df_plot["qobs"], label="Observed Q", linewidth=2)
    ax.plot(df_plot.index, df_plot["qsim"], label="LSTM Sim Q", linewidth=2)

    ax.set_title(f"{basin_id} | NSE={nse:.2f} | KGE={kge:.2f}", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow (Normalized)")
    ax.grid(True)
    ax.legend()

    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    save_path = FIGURES_DIR / f"TS_{basin_id}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# =========================
# Save metrics table
# =========================
df_metrics = pd.DataFrame(metrics, columns=["BasinID", "NSE", "KGE"])
# df_metrics.to_csv(FIGURES_DIR / "LSTM_Performance_Summary.csv", index=False)

print("\n🎯 DONE!")
print(f"📂 Time-series plots saved → {FIGURES_DIR}")
print("📊 Metrics table → LSTM_Performance_Summary.csv")



























































# # import sys
# # # Manually add the Scripts directory to the Python path
# # sys.path.append(r"C:\Users\hdagne1\Box\NRT_Project_2025Fall\Habtamu\HydroAuditToolFrameowrk") 

# # import pickle
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from pathlib import Path
# # from scipy.stats import gaussian_kde
# # # Now the import should work using the Scripts prefix
# # from Scripts.metrics import calc_nse

# # # --- Configuration (Adjust Paths) ---
# # BASE_DIR = Path(r"C:\Users\hdagne1\Box\NRT_Project_2025Fall\Habtamu\HydroAuditToolFrameowrk")
# # BASE_RUN_DIR = BASE_DIR / "src" / "runs"
# # STATS_DIR = BASE_DIR / "stats"
# # FIGURES_DIR = BASE_DIR / "figures"

# # FIGURES_DIR.mkdir(exist_ok=True)
# # STATS_DIR.mkdir(exist_ok=True)

# # # --- Utility Functions (Translated from plotutils.py and MATLAB) ---

# # # MATLAB's plot_main.m expects 5 models. We need 5 columns of data.
# # MODEL_NAMES = ['SAC-SMA', 'NWM', 'Global LSTM (no statics)', 'Global LSTM (with statics)', 'PUB LSTM']
# # N_MODELS = len(MODEL_NAMES)

# # # STAT NAMES (from plot_main.m)
# # STAT_NAMES = ['Nash Sutcliffe Efficiency', 'Fractional Bias', 
# #               'Stdandard Deviation Ratio', '95th Percentile Difference']

# # # Axis limits (from plot_main.m)
# # AXIS_LIMS = np.array([[-1, 1], [-2, 1], [0, 2.5], [-1, 1]])
# # OPTIMAL = [1, 0, 1, 0] # Optimal values for NSE, Bias, Std Rat, 95% Diff

# # # Colors (Approximation of MATLAB's grab_plot_colors or using distinct matplotlib colors)
# # COLORS = plt.cm.get_cmap('Dark2', N_MODELS).colors # Using a distinct color map


# # def calculate_metrics(qobs: np.ndarray, qsim: np.ndarray) -> dict:
# #     """Calculates all required CAMELS metrics for a single basin."""
# #     qobs = qobs.flatten()
# #     qsim = qsim.flatten()
# #     qobs_mean = np.mean(qobs)
# #     qobs_std = np.std(qobs)
# #     qsim_std = np.std(qsim)
    
# #     # 1. Nash Sutcliffe Efficiency: Use the verified function
# #     try:
# #         nse = calc_nse(qobs, qsim)
# #     except RuntimeError:
# #         # NSE is undefined if obs is constant (denominator=0 in calc_nse)
# #         nse = np.nan

# #     # 2. Fractional Bias
# #     bias = (np.mean(qsim) - qobs_mean) / qobs_mean if qobs_mean != 0 else np.nan

# #     # 3. Stdandard Deviation Ratio
# #     std_rat = qsim_std / qobs_std if qobs_std != 0 else np.nan

# #     # 4. 95th Percentile Difference (Matches MATLAB logic: (Qobs_95 - Qsim_95) / Qobs_95)
# #     qobs_95 = np.percentile(qobs, 95)
# #     qsim_95 = np.percentile(qsim, 95)
# #     q95_diff = (qobs_95 - qsim_95) / qobs_95 if qobs_95 != 0 else np.nan

# #     # FIX: Ensure these keys EXACTLY match the STAT_NAMES list for aggregation
# #     return {
# #         'Nash Sutcliffe Efficiency': nse,
# #         'Fractional Bias': bias,
# #         'Stdandard Deviation Ratio': std_rat,
# #         '95th Percentile Difference': q95_diff
# #     }


# # def load_and_aggregate_results():
# #     """Loads your LSTM results and combines them with placeholder/benchmark data."""
# #     all_basin_data = {}  # {basin_id: {seed: {metrics}}}
    
# #     # --- Load Your Global LSTM (no statics) Results ---
# #     run_dirs = sorted(BASE_RUN_DIR.glob("run_*_seed*"))
    
# #     if not run_dirs:
# #         print("Error: No 'run_...' directories found. Cannot run plotting.")
# #         return None
        
# #     for run_dir in run_dirs:
# #         seed = int(run_dir.name.split('_seed')[-1])
# #         result_file = run_dir / f"lstm_no_static_seed{seed}.p"
        
# #         if not result_file.exists():
# #             continue

# #         with open(result_file, 'rb') as fp:
# #             results = pickle.load(fp)

# #         for basin_id, df in results.items():
# #             if basin_id not in all_basin_data:
# #                 all_basin_data[basin_id] = {}
# #             metrics = calculate_metrics(df['qobs'].values, df['qsim'].values)
# #             all_basin_data[basin_id][seed] = metrics

# #     # Aggregate metrics across seeds (using MEDIAN as in standard practice)
# #     aggregated_lstm_data = {}
# #     for basin_id, seed_data in all_basin_data.items():
# #         if not seed_data: continue
        
# #         basin_metrics = {}
# #         for metric_name in STAT_NAMES:
# #             # FIX: This aggregation now works because calculate_metrics uses the correct keys
# #             all_values = [data[metric_name] for data in seed_data.values()]
# #             basin_metrics[metric_name] = np.nanmedian(all_values)
        
# #         aggregated_lstm_data[basin_id] = basin_metrics

# #     # Convert to DataFrame
# #     df_lstm = pd.DataFrame.from_dict(aggregated_lstm_data, orient='index')
# #     df_lstm.index.name = 'BasinID'
    
# #     # --- Load or Create Benchmark Data ---
# #     basin_ids = df_lstm.index.values
# #     N_BASINS = len(basin_ids)
    
# #     # Initialize the final STATS array (nStats, nBasins, nModels)
# #     stats_array = np.full((len(STAT_NAMES), N_BASINS, N_MODELS), np.nan)
    
# #     np.random.seed(42) # Consistent randomness for placeholders
    
# #     for i, basin_id in enumerate(basin_ids):
# #         # 3. Global LSTM (no statics) - YOUR DATA (Model index 2)
# #         for s, stat in enumerate(STAT_NAMES):
# #             stats_array[s, i, 2] = df_lstm.loc[basin_id, stat]

# #         # Benchmarks (Placeholders)
# #         # We must ensure placeholder NSE values are NOT identical to avoid LinAlgError
# #         nse_lstm = df_lstm.loc[basin_id, 'Nash Sutcliffe Efficiency']
        
# #         # SAC-SMA (Model 0) - NSE slightly worse than LSTM
# #         stats_array[0, i, 0] = nse_lstm - np.random.uniform(0.05, 0.1)
        
# #         # NWM (Model 1) - NSE slightly worse than SAC-SMA
# #         stats_array[0, i, 1] = stats_array[0, i, 0] - np.random.uniform(0.05, 0.1)
        
# #         # Global LSTM (with statics) (Model 3) - NSE slightly better than your model
# #         stats_array[0, i, 3] = nse_lstm + np.random.uniform(0, 0.02)
        
# #         # PUB LSTM (Model 4) - NSE comparable to your model
# #         stats_array[0, i, 4] = nse_lstm + np.random.uniform(-0.01, 0.01)

# #         # Apply random data to other metrics (Bias, Std, 95%) 
# #         for s in [1, 2, 3]:
# #             stats_array[s, i, 0:5] = np.random.uniform(AXIS_LIMS[s, 0] / 2, AXIS_LIMS[s, 1] / 2, size=5)
# #             # Ensure Bias is centered around 0, Std Rat around 1
# #             if s == 2:
# #                  stats_array[s, i, 0:5] = np.random.uniform(0.8, 1.2, size=5)

# #     # Clip NSE to [-1, 1] 
# #     stats_array[0, :, :] = np.clip(stats_array[0, :, :], -1, 1)
    
# #     return stats_array

# # # --- Step 3: Plotting Functions (Translated from MATLAB) ---
# # def plot_pdfs_cdfs(stats_array):
# #     for s, stat_name in enumerate(STAT_NAMES):
# #         plt.figure(figsize=(18, 6))
# #         plt.suptitle(f"Frequencies of {stat_name} Values over {stats_array.shape[1]} Basins", fontsize=20)

# #         # PDF subplot
# #         plt.subplot(1, 3, (1, 2))
# #         x_min, x_max = AXIS_LIMS[s, :]
# #         x_linspace = np.linspace(x_min, x_max, 100)

# #         for m in range(N_MODELS):
# #             data = stats_array[s, :, m]
# #             data = data[~np.isnan(data)]

# #             if len(data) > 1:
# #                 if np.std(data) > 1e-6:
# #                     kde = gaussian_kde(data)
# #                     f = kde(x_linspace)
# #                     plt.plot(x_linspace, f, label=MODEL_NAMES[m], color=COLORS[m], linewidth=3)
# #                 else:
# #                     hist, bins = np.histogram(data, bins=3, range=(x_min, x_max), density=True)
# #                     centers = (bins[:-1] + bins[1:]) / 2
# #                     plt.plot(centers, hist, linestyle='--', marker='o', linewidth=2,
# #                              label=MODEL_NAMES[m], color=COLORS[m])

# #         plt.xlim(x_min, x_max)
# #         plt.grid(True)
# #         plt.tick_params(labelsize=14)
# #         plt.ylabel('f(x)', fontsize=18)
# #         plt.xlabel(stat_name, fontsize=18)
# #         plt.legend(loc='upper left', fontsize=12)

# #         # CDF subplot
# #         plt.subplot(1, 3, 3)
# #         for m in range(N_MODELS):
# #             data = stats_array[s, :, m]
# #             data = data[~np.isnan(data)]

# #             if len(data) > 1:
# #                 xs = np.sort(data)
# #                 ys = np.arange(1, len(xs) + 1) / len(xs)
# #                 plt.plot(xs, ys, label=MODEL_NAMES[m], color=COLORS[m], linewidth=3)

# #         plt.xlim(x_min, x_max)
# #         plt.grid(True)
# #         plt.tick_params(labelsize=14)
# #         plt.title('Cumulative Distribution', fontsize=16)

# #         figname = FIGURES_DIR / f"frequencies_{stat_name.replace(' ', '_')}.png"
# #         plt.savefig(figname, bbox_inches='tight')
# #         plt.show()
# #         plt.close()


# # def plot_scatter_comparison(stats_array):
# #     for s, stat_name in enumerate(STAT_NAMES):
# #         plt.figure(figsize=(8, 8))

# #         x_min, x_max = AXIS_LIMS[s, :]

# #         plt.plot([x_min, x_max], [x_min, x_max], 'k--', linewidth=2)

# #         valid_mask = ~np.isnan(stats_array[s, :, 2])
# #         x_data = stats_array[s, valid_mask, 3]
# #         y_sac = stats_array[s, valid_mask, 0]
# #         y_nwm = stats_array[s, valid_mask, 1]
# #         y_nostat = stats_array[s, valid_mask, 2]
# #         y_pub = stats_array[s, valid_mask, 4]

# #         h1 = plt.plot(x_data, y_sac, 'o', markersize=7, color=COLORS[0], alpha=0.7, label=MODEL_NAMES[0])[0]
# #         h2 = plt.plot(x_data, y_nwm, '+', markersize=7, color=COLORS[1], alpha=0.8, label=MODEL_NAMES[1])[0]
# #         h3 = plt.plot(x_data, y_nostat, '^', markersize=7, color=COLORS[2], alpha=0.8, label=MODEL_NAMES[2])[0]
# #         h4 = plt.plot(x_data, y_pub, 'x', markersize=7, color=COLORS[4], alpha=0.8, label=MODEL_NAMES[4])[0]

# #         plt.grid(True)
# #         plt.tick_params(labelsize=14)
# #         plt.xlim(x_min, x_max)
# #         plt.ylim(x_min, x_max)

# #         plt.xlabel(MODEL_NAMES[3], fontsize=18)
# #         plt.ylabel(stat_name, fontsize=18)
# #         plt.title(stat_name, fontsize=20)

# #         plt.legend(handles=[h1, h2, h3, h4], loc='upper left', fontsize=12)

# #         figname = FIGURES_DIR / f"global_lstm_scatters_{stat_name.replace(' ', '_')}.png"
# #         plt.savefig(figname, bbox_inches='tight')
# #         plt.show()
# #         plt.close()

# # # --- Main Execution ---

# # def main():
# #     # 1. Load and prepare all data 
# #     stats_array = load_and_aggregate_results()
    
# #     if stats_array is None:
# #         print("Script failed to load data. Please ensure BASE_RUN_DIR is correct.")
# #         return

# #     # 2. Print LaTeX summary table
# #     print("\n--- Summary Table (Mean and Median of All Basins) ---")
    
# #     for s, stat_name in enumerate(STAT_NAMES):
# #         print(f"\n\\textbf{{{stat_name}:}} & & & & ")
# #         for m, model_name in enumerate(MODEL_NAMES):
# #             data = stats_array[s, :, m]
# #             data = stats_array[s, ~np.isnan(stats_array[s, :, m]), m]

# #             data = data[~np.isnan(data)]
            
# #             # Skip if no valid data
# #             if len(data) == 0:
# #                 ens_median = ens_mean = ens_min = ens_max = np.nan
# #             else:
# #                 ens_median = np.median(data)
# #                 ens_mean = np.mean(data)
# #                 ens_min = np.min(data)
# #                 ens_max = np.max(data)
            
# #             print(f"\\hspace{{1em}}{model_name}: & {ens_median:3.2f} & {ens_mean:3.2f} & {ens_min:3.2f} & {ens_max:3.2f}")

# #     print("\n--- Generating Plots ---")
    
# #     # 3. Plot PDFs and CDFs (Figure 1 in the paper)
# #     plot_pdfs_cdfs(stats_array)

# #     # 4. Plot Comparative Scatterplots (Figure 2 in the paper)
# #     plot_scatter_comparison(stats_array)

# #     print(f"\n✅ Plots saved successfully to the '{FIGURES_DIR.name}' folder!")
# #     print("\n\t*WARNING*: Benchmark models (SAC-SMA, NWM, etc.) use placeholder data.")


# # if __name__ == "__main__":
# #     main()


