#!/usr/bin/env python3
"""
graphgitcommits - Visualize git commits by user and time.

Usage:
    graphgitcommits                           # All commits, weekly
    graphgitcommits --period 1y               # Last 1 year
    graphgitcommits --period 6m               # Last 6 months
    graphgitcommits --interval month          # Aggregate by month
    graphgitcommits -p 2y -i quarter          # Last 2 years, by quarter

Run from any git repository to generate commit visualizations.
"""

import argparse
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

def get_repo_name() -> str:
    """Get the name of the current git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Extract just the directory name from the full path
        repo_path = result.stdout.strip()
        return repo_path.split("/")[-1]
    except subprocess.CalledProcessError:
        return "repo"

def parse_period(period: str | None) -> datetime | None:
    """Parse period string like '1y', '6m', '30d' into a cutoff datetime."""
    if not period:
        return None
    period = period.lower().strip()
    now = datetime.now()

    if period.endswith("y"):
        years = int(period[:-1])
        return now - timedelta(days=years * 365)
    elif period.endswith("m"):
        months = int(period[:-1])
        return now - timedelta(days=months * 30)
    elif period.endswith("w"):
        weeks = int(period[:-1])
        return now - timedelta(weeks=weeks)
    elif period.endswith("d"):
        days = int(period[:-1])
        return now - timedelta(days=days)
    else:
        raise ValueError(f"Invalid period format: {period}. Use format like '1y', '6m', '30d'")


def to_ascii(s: str) -> str:
    """Normalize string to ASCII-like for matching (handles accented characters)."""
    s = s.lower()
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ý": "y",
        "ð": "d", "þ": "th", "æ": "ae", "ö": "o",
        "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u",
        "ä": "a", "ë": "e", "ï": "i", "ü": "u",
        "ø": "o", "å": "a", "ñ": "n", "ç": "c",
    }
    for char, replacement in replacements.items():
        s = s.replace(char, replacement)
    return s


def get_first_name(author: str) -> str:
    """Extract and return the first name from an author string.

    Handles:
    - Regular names: 'John Smith' -> 'John'
    - Usernames with dots: 'john.smith' -> 'john'
    """
    author = author.strip()

    # If it contains a dot but no space, it's likely a username like 'john.smith'
    if "." in author and " " not in author:
        return author.split(".")[0]

    return author.split()[0] if author.split() else author


def build_name_map(raw_commits: List[Tuple[str, datetime, int]]) -> Dict[str, str]:
    """
    Build a dynamic mapping from ASCII-normalized first names to canonical display names.
    Picks the most common variation as the canonical form, preferring properly accented versions.
    """
    # Count occurrences of each (ascii_name, original_first_name) pair
    name_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for author, _, _ in raw_commits:
        first_name = get_first_name(author)
        ascii_name = to_ascii(first_name)
        name_counts[ascii_name][first_name] += 1

    # For each ASCII name, pick the best canonical form
    name_map: Dict[str, str] = {}
    for ascii_name, variations in name_counts.items():

        def score(name: str) -> Tuple[int, int, bool]:
            count = variations[name]
            has_accents = any(c in name for c in "áéíóúýðþæöÁÉÍÓÚÝÐÞÆÖ")
            is_capitalized = name[0].isupper() if name else False
            return (count, int(has_accents), is_capitalized)

        best_name = max(variations.keys(), key=score)
        # Ensure proper capitalization
        name_map[ascii_name] = best_name.capitalize() if best_name.islower() else best_name

    return name_map


def get_git_commits(since: datetime | None = None) -> List[Tuple[str, datetime, int]]:
    """Get git commits with author, date, and lines changed, optionally filtered by date."""
    # Use a format that includes commit hash as separator for numstat parsing
    cmd = ["git", "log", "--pretty=format:COMMIT|%an|%ad", "--date=iso", "--numstat", "--all"]

    if since:
        cmd.append(f"--since={since.strftime('%Y-%m-%d')}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse commits with their numstat data
    raw_commits: List[Tuple[str, datetime, int]] = []
    current_author = None
    current_date = None
    current_lines = 0

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        if line.startswith("COMMIT|"):
            # Save previous commit if exists
            if current_author is not None and current_date is not None:
                if since is None or current_date >= since:
                    raw_commits.append((current_author, current_date, current_lines))
            # Parse new commit
            parts = line.split("|", 2)
            current_author = parts[1]
            date_str = parts[2]
            current_date = datetime.fromisoformat(date_str.rsplit(" ", 1)[0])
            current_lines = 0
        else:
            # This is a numstat line: additions<tab>deletions<tab>filename
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    # Handle binary files which show '-' instead of numbers
                    additions = int(parts[0]) if parts[0] != "-" else 0
                    deletions = int(parts[1]) if parts[1] != "-" else 0
                    current_lines += additions + deletions
                except ValueError:
                    pass

    # Don't forget the last commit
    if current_author is not None and current_date is not None:
        if since is None or current_date >= since:
            raw_commits.append((current_author, current_date, current_lines))

    # Build dynamic name mapping
    name_map = build_name_map(raw_commits)

    # Second pass: normalize names
    commits: List[Tuple[str, datetime, int]] = []
    for author, date, lines_changed in raw_commits:
        first_name = get_first_name(author)
        ascii_name = to_ascii(first_name)
        canonical_name = name_map.get(ascii_name, first_name.capitalize())
        commits.append((canonical_name, date, lines_changed))

    return commits


def aggregate_commits(
    commits: List[Tuple[str, datetime, int]], interval: str = "week"
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """Aggregate commits by user and time interval.

    Args:
        commits: List of (author, date, lines_changed) tuples
        interval: One of 'week', 'month', 'quarter', 'year'

    Returns:
        Tuple of (commit_counts, lines_changed) aggregated dictionaries
    """
    commit_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lines_changed: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for author, date, lines in commits:
        if interval == "week":
            # Get the start of the week (Monday)
            period_start = date - pd.Timedelta(days=date.weekday())
            period_key = period_start.strftime("%Y-%m-%d")
        elif interval == "month":
            period_key = date.strftime("%Y-%m")
        elif interval == "quarter":
            quarter = (date.month - 1) // 3 + 1
            period_key = f"{date.year}-Q{quarter}"
        elif interval == "year":
            period_key = str(date.year)
        else:
            raise ValueError(
                f"Invalid interval: {interval}. Use 'week', 'month', 'quarter', or 'year'"
            )

        commit_counts[period_key][author] += 1
        lines_changed[period_key][author] += lines

    return commit_counts, lines_changed


def create_visualization(
    aggregated_commits: Dict[str, Dict[str, int]],
    aggregated_lines: Dict[str, Dict[str, int]],
    commits: List[Tuple[str, datetime, int]],
    interval: str = "week",
    period: str | None = None,
    repo_name: str = "repo",
) -> None:
    """Create a combined visualization with 4 tiles: commits, lines, trend, and stats table."""
    # Convert to DataFrames for easier plotting
    df = pd.DataFrame(aggregated_commits).T.fillna(0)
    df_lines = pd.DataFrame(aggregated_lines).T.fillna(0)

    # Sort index appropriately based on interval
    if interval == "week":
        df.index = pd.to_datetime(df.index)
        df_lines.index = pd.to_datetime(df_lines.index)
    elif interval == "month":
        df.index = pd.to_datetime(df.index + "-01")
        df_lines.index = pd.to_datetime(df_lines.index + "-01")
    elif interval == "quarter":

        def parse_quarter(q: str) -> datetime:
            year, qtr = q.split("-Q")
            month = (int(qtr) - 1) * 3 + 1
            return datetime(int(year), month, 1)

        df.index = pd.to_datetime([parse_quarter(q) for q in df.index])
        df_lines.index = pd.to_datetime([parse_quarter(q) for q in df_lines.index])
    elif interval == "year":
        df.index = pd.to_datetime(df.index + "-01-01")
        df_lines.index = pd.to_datetime(df_lines.index + "-01-01")

    df = df.sort_index()
    df_lines = df_lines.sort_index()

    # Sort columns by total commits (highest first)
    author_totals = df.sum().sort_values(ascending=False)
    author_lines_totals = df_lines.sum()
    df = df[author_totals.index]
    df_lines = df_lines[author_totals.index]  # Same order as commits

    # Create consistent color map for authors
    cmap = plt.cm.get_cmap("tab20")
    author_colors = {author: cmap(i % 20) for i, author in enumerate(author_totals.index)}
    color_list = [author_colors[author] for author in df.columns]

    # Calculate average time between commits per author
    author_commit_times: Dict[str, List[datetime]] = defaultdict(list)
    for author, date, _ in commits:
        author_commit_times[author].append(date)

    avg_time_between: Dict[str, str] = {}
    for author in author_totals.index:
        times = sorted(author_commit_times.get(author, []))
        if len(times) > 1:
            deltas = [(times[i] - times[i - 1]).days for i in range(1, len(times))]
            avg_days = sum(deltas) / len(deltas)
            if avg_days < 1:
                avg_time_between[author] = f"{avg_days * 24:.1f}h"
            elif avg_days < 7:
                avg_time_between[author] = f"{avg_days:.1f}d"
            else:
                avg_time_between[author] = f"{avg_days / 7:.1f}w"
        else:
            avg_time_between[author] = "N/A"

    # Create 2x2 figure with widescreen ratio
    fig = plt.figure(figsize=(24, 12))

    # Use GridSpec for better control
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

    interval_label = interval.capitalize()

    # Format x-axis dates based on interval
    if interval == "week":
        date_format = "%Y-%m-%d"
    elif interval == "month":
        date_format = "%Y-%m"
    elif interval == "quarter":
        date_format = "%Y-Q"
    else:  # year
        date_format = "%Y"

    # Create x-axis labels
    if interval == "quarter":
        labels = [f"{d.year}-Q{(d.month - 1) // 3 + 1}" for d in df.index]
    else:
        labels = [d.strftime(date_format) for d in df.index]

    n = max(1, len(df) // 20)  # Show roughly 20 labels

    # ===== Tile 1: Commit Count (top-left) =====
    ax1 = fig.add_subplot(gs[0, 0])
    df.plot(kind="bar", stacked=True, ax=ax1, width=0.8, legend=False, color=color_list)
    ax1.set_title(f"Commits by {interval_label}", fontsize=14, fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Commits", fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for i, label in enumerate(ax1.xaxis.get_ticklabels()):
        if i % n != 0:
            label.set_visible(False)

    # ===== Tile 2: Lines Changed (top-right) =====
    ax2 = fig.add_subplot(gs[0, 1])
    df_lines.plot(kind="bar", stacked=True, ax=ax2, width=0.8, legend=False, color=color_list)
    ax2.set_title(f"Lines Changed by {interval_label}", fontsize=14, fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("Lines", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for i, label in enumerate(ax2.xaxis.get_ticklabels()):
        if i % n != 0:
            label.set_visible(False)

    # ===== Tile 3: Commits Trend (bottom-left) =====
    ax3 = fig.add_subplot(gs[1, 0])
    for author in df.columns:
        ax3.plot(
            df.index,
            df[author],
            marker="o",
            markersize=3,
            linewidth=2,
            label=author,
            color=author_colors[author],
        )
    ax3.set_title("Commits Trend", fontsize=14, fontweight="bold")
    ax3.set_xlabel(f"{interval_label}", fontsize=10)
    ax3.set_ylabel("Commits", fontsize=10)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    if interval == "year":
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    elif interval == "quarter":
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    elif interval == "month":
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    # ===== Tile 4: Stats Table (bottom-right) =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    # Calculate number of months in the date range for avg lines/month
    date_range_days = (df.index.max() - df.index.min()).days
    num_months = max(1, date_range_days / 30.44)  # Average days per month

    # Build table data with color indicator
    table_data = []
    for author in author_totals.index:
        commit_count = int(author_totals[author])
        line_count = int(author_lines_totals[author])
        avg_lines = line_count // commit_count if commit_count > 0 else 0
        avg_lines_month = int(line_count / num_months)
        avg_time = avg_time_between[author]
        table_data.append(
            ["", author, f"{commit_count:,}", f"{line_count:,}", f"{avg_lines:,}", f"{avg_lines_month:,}", avg_time]
        )

    # Add totals row
    total_commits = int(df.sum().sum())
    total_lines = int(df_lines.sum().sum())
    avg_lines_total = total_lines // total_commits if total_commits > 0 else 0
    avg_lines_month_total = int(total_lines / num_months)
    table_data.append(
        [
            "",
            "TOTAL",
            f"{total_commits:,}",
            f"{total_lines:,}",
            f"{avg_lines_total:,}",
            f"{avg_lines_month_total:,}",
            "—",
        ]
    )

    col_labels = ["", "Author", "Commits", "Lines", "Lines/Commit", "Lines/Month", "Time Between"]

    table = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=[0.04, 0.19, 0.11, 0.14, 0.16, 0.16, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.5)

    # Style the header row
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Set column alignments: author left-aligned, numbers right-aligned
    for row_idx in range(len(table_data) + 1):  # +1 for header
        # Column 1 (Author) - left align
        table[(row_idx, 1)].get_text().set_ha("left")
        # Columns 2-6 (numbers) - right align
        for col_idx in range(2, 7):
            table[(row_idx, col_idx)].get_text().set_ha("right")

    # Style author rows with their colors in the first column
    for row_idx, author in enumerate(author_totals.index, start=1):
        table[(row_idx, 0)].set_facecolor(author_colors[author])

    # Style the totals row
    for i in range(len(col_labels)):
        table[(len(table_data), i)].set_facecolor("#E2EFDA")
        table[(len(table_data), i)].set_text_props(fontweight="bold")

    ax4.set_title("Contributor Statistics", fontsize=14, fontweight="bold", pad=20)

    # Add overall title
    period_text = f" ({period})" if period else " (all time)"
    fig.suptitle(f"{repo_name} - Git Activity{period_text}", fontsize=18, fontweight="bold", y=0.98)

    # Build filename with parameters
    period_str = f"_{period}" if period else "_all"
    filename_base = f"{repo_name}_commits_{interval}{period_str}"

    # Save the combined figure
    output_file = f"{filename_base}_dashboard.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Dashboard saved to {output_file}")

    # Print statistics
    print("\n📊 Statistics:")
    print(f"Total {interval}s analyzed: {len(df)}")
    print(
        f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
    )
    print("\nTotal commits and lines by author (sorted by commits):")
    for author in author_totals.index:
        commit_count = int(author_totals[author])
        lines = int(author_lines_totals[author])
        avg_lines = lines // commit_count if commit_count > 0 else 0
        print(f"  {author}: {commit_count:,} commits, {lines:,} lines ({avg_lines:,} avg lines/commit)")
    print(f"\nGrand total: {total_commits:,} commits, {total_lines:,} lines changed")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize git commits by user and time interval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  graphgitcommits                    # All commits, aggregated by week
  graphgitcommits -p 1y              # Last 1 year
  graphgitcommits -p 6m -i month     # Last 6 months, aggregated by month
  graphgitcommits -p 2y -i quarter   # Last 2 years, aggregated by quarter
  graphgitcommits -i year            # All time, aggregated by year

Period formats:
  1y, 2y     - Years
  6m, 12m    - Months
  4w, 8w     - Weeks
  30d, 90d   - Days
        """,
    )
    parser.add_argument(
        "--period",
        "-p",
        type=str,
        help="Time period to analyze (e.g., 1y, 6m, 3m, 30d, 2w)",
        default=None,
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=str,
        choices=["week", "month", "quarter", "year"],
        help="Aggregation interval (default: week)",
        default="week",
    )
    args = parser.parse_args()

    since = parse_period(args.period)
    period_desc = f" (last {args.period})" if args.period else ""
    repo_name = get_repo_name()

    print(f"Fetching git commits for '{repo_name}'{period_desc}...")
    try:
        commits = get_git_commits(since)
    except subprocess.CalledProcessError:
        print("❌ Error: Not a git repository or git is not installed.")
        print("   Run this command from within a git repository.")
        raise SystemExit(1)

    if not commits:
        print("❌ No commits found in this repository.")
        raise SystemExit(1)

    print(f"✓ Found {len(commits):,} commits")

    print(f"\nAggregating by {args.interval}...")
    aggregated_commits, aggregated_lines = aggregate_commits(commits, args.interval)
    print(f"✓ Aggregated into {len(aggregated_commits)} {args.interval}s")

    print("\nCreating visualizations...")
    create_visualization(aggregated_commits, aggregated_lines, commits, args.interval, args.period, repo_name)

    print("\n✅ Done! Check the generated PNG file.")


if __name__ == "__main__":
    main()
