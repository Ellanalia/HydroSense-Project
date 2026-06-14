"""
visualization.py – HydroSense-Kenya
ICS 2207 Scientific Computing | Level 4 & 5
Reusable scientific plotting functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

ZONE_COLORS = {'Zone_A': '#e07b39', 'Zone_B': '#3a7ebf', 'Zone_C': '#5aab61'}
ZONE_CROPS  = {'Zone_A': 'Tomato', 'Zone_B': 'Kale', 'Zone_C': 'Maize'}
STRESS_COLORS = {'OK': '#5aab61', 'IRRIGATE': '#f0c060', 'STRESS': '#cc3333', 'UNKNOWN': '#aaaaaa'}

plt.rcParams.update({
    'figure.dpi': 120, 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11
})


def plot_rainfall_et(weather_df, save_path=None):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(weather_df['date'], weather_df['rainfall_mm'], color='steelblue', alpha=0.6, label='Rainfall (mm)')
    ax2.plot(weather_df['date'], weather_df['et_mm'], color='firebrick', lw=2, marker='o', ms=4, label='ET (mm/day)')
    ax1.set_xlabel('Date'); ax1.set_ylabel('Rainfall (mm/day)', color='steelblue')
    ax2.set_ylabel('ET (mm/day)', color='firebrick')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.xticks(rotation=30)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc='upper left')
    plt.title('Daily Rainfall and Estimated Evapotranspiration – March 2026', fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_soil_moisture_zones(soil_df, params_df, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    for zone_id, group in soil_df.groupby('zone_id'):
        group = group.sort_values('timestamp')
        ax.plot(group['timestamp'], group['soil_moisture_pct'],
                color=ZONE_COLORS[zone_id], lw=2, marker='o', ms=3,
                label=f"{zone_id} ({ZONE_CROPS[zone_id]})")
    for _, row in params_df.iterrows():
        ax.axhline(row['min_moisture_pct'], color=ZONE_COLORS[row['zone_id']], ls=':', alpha=0.5, lw=1)
    ax.set_xlabel('Date'); ax.set_ylabel('Soil Moisture (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.xticks(rotation=30); ax.legend()
    plt.title('Soil Moisture by Farm Zone – March 2026\n(Dotted = stress threshold)', fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_simulation_results(euler_S, rk_S, dates_extended, zone_id, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates_extended, euler_S, color='steelblue', lw=2, label='Euler method')
    ax.plot(dates_extended, rk_S,    color='firebrick', lw=2, ls='--', label='Runge-Kutta (RK4)')
    ax.set_xlabel('Date'); ax.set_ylabel('Soil Moisture (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.xticks(rotation=30); ax.legend()
    plt.title(f'Soil Moisture Simulation – {zone_id} ({ZONE_CROPS.get(zone_id, "")})', fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_monte_carlo(summary, dates_extended, zone_id, min_moisture, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(dates_extended, summary['p5'], summary['p95'], alpha=0.2, color='steelblue', label='5–95th pct')
    ax.fill_between(dates_extended, summary['p25'], summary['p75'], alpha=0.35, color='steelblue', label='25–75th pct')
    ax.plot(dates_extended, summary['mean'], color='steelblue', lw=2, label='Mean')
    ax.axhline(min_moisture, color='red', ls='--', lw=1.5, label=f'Stress threshold ({min_moisture}%)')
    ax.set_xlabel('Date'); ax.set_ylabel('Soil Moisture (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=30); ax.legend()
    plt.title(f'Monte Carlo Uncertainty – {zone_id} (1000 rainfall scenarios)', fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
