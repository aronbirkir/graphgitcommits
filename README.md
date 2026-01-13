# graphgitcommits

Visualize git commits by user and time, with flexible aggregation intervals.

## Installation

### Run directly with uvx (no installation needed)

```bash
# From any git repository:
uvx --from git+https://github.com/aronbirkir/graphgitcommits graphgitcommits -p 1y -i month
```

### Install globally with uv

```bash
uv tool install git+https://github.com/aronbirkir/graphgitcommits
```

Then run from any git repository:

```bash
graphgitcommits -p 1y -i month
```

## Usage

Run from within any git repository:

```bash
# All commits, aggregated by week (default)
graphgitcommits

# Last 1 year, weekly
graphgitcommits --period 1y

# Last 6 months, aggregated by month
graphgitcommits -p 6m -i month

# Last 2 years, aggregated by quarter
graphgitcommits -p 2y -i quarter

# All time, aggregated by year
graphgitcommits -i year
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--period` | `-p` | Time period to analyze (e.g., `1y`, `6m`, `3m`, `30d`, `2w`) |
| `--interval` | `-i` | Aggregation interval: `week`, `month`, `quarter`, `year` |

### Period formats

- `1y`, `2y` - Years
- `6m`, `12m` - Months  
- `4w`, `8w` - Weeks
- `30d`, `90d` - Days

## Output

The tool generates two PNG files:

1. `git_commits_{interval}_{period}.png` - Stacked bar chart
2. `git_commits_{interval}_{period}_trend.png` - Line chart showing trends

## Features

- **Smart name consolidation**: Automatically groups authors by first name, handling:
  - Different name formats (e.g., "John Smith" and "john.smith")
  - Accented characters (e.g., "Jón" and "Jon" are grouped together)
  - Prefers the most common spelling with proper accents

- **Flexible time periods**: Analyze any time range from days to years

- **Multiple aggregation levels**: View commits by week, month, quarter, or year

- **Sorted by activity**: Authors are sorted by total commits, highest first

## License

MIT
