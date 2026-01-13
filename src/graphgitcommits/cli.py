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


def build_name_map(raw_commits: List[Tuple[str, datetime]]) -> Dict[str, str]:
    """
    Build a dynamic mapping from ASCII-normalized first names to canonical display names.
    Picks the most common variation as the canonical form, preferring properly accented versions.
    """
    # Count occurrences of each (ascii_name, original_first_name) pair
    name_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for author, _ in raw_commits:
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


def get_git_commits(since: datetime | None = None) -> List[Tuple[str, datetime]]:
    """Get git commits with author and date, optionally filtered by date."""
    cmd = ["git", "log", "--pretty=format:%an|%ad", "--date=iso", "--all"]

    if since:
        cmd.append(f"--since={since.strftime('%Y-%m-%d')}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    # First pass: collect raw commits
    raw_commits: List[Tuple[str, datetime]] = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        author, date_str = line.split("|", 1)
        date = datetime.fromisoformat(date_str.rsplit(" ", 1)[0])
        if since and date < since:
            break
        raw_commits.append((author, date))

    # Build dynamic name mapping
    name_map = build_name_map(raw_commits)

    # Second pass: normalize names
    commits: List[Tuple[str, datetime]] = []
    for author, date in raw_commits:
        first_name = get_first_name(author)
        ascii_name = to_ascii(first_name)
        canonical_name = name_map.get(ascii_name, first_name.capitalize())
        commits.append((canonical_name, date))

    return commits


def aggregate_commits(
    commits: List[Tuple[str, datetime]], interval: str = "week"
) -> Dict[str, Dict[str, int]]:
    """Aggregate commits by user and time interval.

    Args:
        commits: List of (author, date) tuples
        interval: One of 'week', 'month', 'quarter', 'year'
    """
    aggregated: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for author, date in commits:
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

        aggregated[period_key][author] += 1

    return aggregated


def create_visualization(
    aggregated_commits: Dict[str, Dict[str, int]],
    interval: str = "week",
    period: str | None = None,
    repo_name: str = "repo",
) -> None:
    """Create a stacked bar chart of commits by user and time interval."""
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(aggregated_commits).T.fillna(0)

    # Sort index appropriately based on interval
    if interval == "week":
        df.index = pd.to_datetime(df.index)
    elif interval == "month":
        df.index = pd.to_datetime(df.index + "-01")
    elif interval == "quarter":

        def parse_quarter(q: str) -> datetime:
            year, qtr = q.split("-Q")
            month = (int(qtr) - 1) * 3 + 1
            return datetime(int(year), month, 1)

        df.index = pd.to_datetime([parse_quarter(q) for q in df.index])
    elif interval == "year":
        df.index = pd.to_datetime(df.index + "-01-01")

    df = df.sort_index()

    # Sort columns by total commits (highest first)
    author_totals = df.sum().sort_values(ascending=False)
    df = df[author_totals.index]

    # Create legend labels with commit counts
    legend_labels = [
        f"{author} ({int(author_totals[author]):,})" for author in df.columns
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot stacked bar chart
    df.plot(kind="bar", stacked=True, ax=ax, width=0.8)

    # Formatting
    interval_label = interval.capitalize()
    ax.set_title(
        f"{repo_name} - Git Commits by User and {interval_label}", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel(f"{interval_label} Starting", fontsize=12)
    ax.set_ylabel("Number of Commits", fontsize=12)
    ax.legend(legend_labels, title="Authors", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

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

    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Only show every nth label to avoid overcrowding
    n = max(1, len(df) // 30)  # Show roughly 30 labels
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % n != 0:
            label.set_visible(False)

    plt.tight_layout()

    # Build filename with parameters
    period_str = f"_{period}" if period else "_all"
    filename_base = f"{repo_name}_commits_{interval}{period_str}"

    # Save the figure
    output_file = f"{filename_base}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Graph saved to {output_file}")

    # Also create a line chart for trends
    fig2, ax2 = plt.subplots(figsize=(16, 8))

    # Plot each author's line explicitly with proper x values
    for i, author in enumerate(df.columns):
        ax2.plot(
            df.index, df[author], marker="o", markersize=3, linewidth=2, label=legend_labels[i]
        )

    ax2.set_title(
        f"{repo_name} - Git Commits Trend by User and {interval_label}", fontsize=16, fontweight="bold"
    )
    ax2.set_xlabel(f"{interval_label} Starting", fontsize=12)
    ax2.set_ylabel("Number of Commits", fontsize=12)
    ax2.legend(title="Authors", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Format x-axis based on interval
    if interval == "year":
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
    elif interval == "quarter":
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    elif interval == "month":
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        month_interval = max(1, len(df) // 12)
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    else:  # week
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Set proper date limits
    ax2.set_xlim(df.index.min(), df.index.max())

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    output_file2 = f"{filename_base}_trend.png"
    plt.savefig(output_file2, dpi=300, bbox_inches="tight")
    print(f"✓ Trend graph saved to {output_file2}")

    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"Total {interval}s analyzed: {len(df)}")
    print(
        f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
    )
    print(f"\nTotal commits by author (sorted by commits):")
    for author in author_totals.index:
        total = int(author_totals[author])
        print(f"  {author}: {total:,} commits")
    print(f"\nGrand total: {int(df.sum().sum()):,} commits")


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
    aggregated = aggregate_commits(commits, args.interval)
    print(f"✓ Aggregated into {len(aggregated)} {args.interval}s")

    print("\nCreating visualizations...")
    create_visualization(aggregated, args.interval, args.period, repo_name)

    print("\n✅ Done! Check the generated PNG files.")


if __name__ == "__main__":
    main()
