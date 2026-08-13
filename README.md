# NBA_3PT_Dep
Primary focus: Analyzing dependence on 3PT in NBA

Paper in progress

Secondary focus: Behavioral analysis of defenses when reacting to hot-hand-related offensive streaks

Kondur, A., & Shen, W. (2026). Statistical analysis of NBA defensive responses to hot-hand streaks. Journal of Sports Analytics, 12. https://doi.org/10.1177/22150218261458601

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for a detailed walkthrough of what each script does.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Play-by-play comes from the Kaggle `wyattowalsh/basketball` dump. Requires a Kaggle API token at
`$env:USERPROFILE\.kaggle\kaggle.json` (Kaggle account → Settings → Create New Token).

```powershell
pip install kaggle
mkdir data
kaggle datasets download -d wyattowalsh/basketball -f csv/play_by_play.csv -p data
kaggle datasets download -d wyattowalsh/basketball -f csv/game_info.csv -p data
kaggle datasets download -d wyattowalsh/basketball -f csv/game.csv -p data
Get-ChildItem data\*.zip | ForEach-Object { Expand-Archive $_ -DestinationPath data -Force; Remove-Item $_ }
```

SportVU optical tracking logs (2015-16, one `.7z` archive per game) are not on Kaggle. They come from
[linouk23/NBA-Player-Movements](https://github.com/linouk23/NBA-Player-Movements/tree/master/data/2016.NBA.Raw.SportVU.Game.Logs)
— 636 games, ~3.6 GB, all falling inside the 2015-10-01 to 2016-01-31 window the pipeline uses.

The number of archives in `data/game_logs/` sets the sample size, since only games with tracking data
are processed. For a quick end-to-end run, fetch a handful:

```powershell
mkdir data\game_logs
$api = "https://api.github.com/repos/linouk23/NBA-Player-Movements/contents/data/2016.NBA.Raw.SportVU.Game.Logs"
(Invoke-RestMethod $api) | Select-Object -First 10 | ForEach-Object {
  Invoke-WebRequest $_.download_url -OutFile "data\game_logs\$($_.name)"
}
```

For the full set, shallow-clone and move the directory into place:

```powershell
git clone --depth 1 https://github.com/linouk23/NBA-Player-Movements.git ..\npm-tmp
Move-Item ..\npm-tmp\data\2016.NBA.Raw.SportVU.Game.Logs\*.7z data\game_logs\
Remove-Item -Recurse -Force ..\npm-tmp
```

Expected layout:

```
data/
  play_by_play.csv
  game_info.csv
  game.csv
  game_logs/      # .7z tracking archives, flat — do not unpack
  moment_data/    # generated, one CSV per game (~41 MB each)
  temp_logs/      # scratch, created and deleted per archive
```

Leave the `.7z` files compressed. `moment_processing.py` opens each one with `py7zr`, extracts the
JSON to `data/temp_logs/`, converts it, and deletes the scratch directory. Keep the archives flat in
`data/game_logs/` — subdirectories are not handled.

## Running

Run from the repository root; all paths are relative to the working directory.

```powershell
python src\moment_processing.py             # reads game_logs\*.7z -> data\moment_data\*.csv
python src\pbp_shot_processing.py           # -> data\pbp_final.csv
python src\shot_prob_model.py               # -> results\shot_prob_xgb.txt
python src\cf_dep_processing.py             # -> data\cf_dep_player.csv, data\cf_dep_team.csv
python src\experiments\recency_bias.py
python src\experiments\defense_during_streaks.py
python src\experiments\def_metric_heat_reg.py
```

Each stage skips work whose output file already exists. Delete that file, or set `redo = True` in the
script's `__main__` block, to force a rebuild. Results are written to `results/` and `plots/`.
