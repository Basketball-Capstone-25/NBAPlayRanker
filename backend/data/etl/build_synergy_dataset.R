#!/usr/bin/env Rscript

cat("Running build_synergy_dataset.R\n")
cat("Working directory:", getwd(), "\n\n")

# ---- packages ----

needed <- c("hoopR", "dplyr", "purrr", "readr", "vroom")
new_pkgs <- needed[!(needed %in% rownames(installed.packages()))]

if (length(new_pkgs) > 0) {
  install.packages(new_pkgs, repos = "https://cloud.r-project.org")
}

invisible(lapply(needed, library, character.only = TRUE))

# ---- basic config ----

PLAYER_OR_TEAM <- "P"  # "P" = players, "T" = teams

SEASONS <- sprintf("%d-%02d", 2019:2024, 20:25)   # "2019-20" ... "2024-25"

PLAYTYPES <- c(
  "Isolation", "Transition", "PRBallHandler", "PRRollman", "Postup",
  "Spotup", "Handoff", "Cut", "OffScreen", "OffRebound", "Misc"
)

SIDES <- c("Offensive", "Defensive")
SEASON_TYPES <- c("Regular Season", "Playoffs")
PER_MODE <- "Totals"

# small random delay between API calls so we do not hammer the endpoint
SLEEP_MIN <- 0.6
SLEEP_MAX <- 1.2

# paths relative to the backend folder
DATA_DIR  <- "data"
CHUNK_DIR <- file.path(DATA_DIR, "synergy_chunks")

FINAL_CSV <- ifelse(
  PLAYER_OR_TEAM == "P",
  file.path(DATA_DIR, "synergy_playtypes_2019_2025_players.csv"),
  file.path(DATA_DIR, "synergy_playtypes_2019_2025_teams.csv")
)

dir.create(CHUNK_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- column layout ----

expected_num <- c(
  "PPP", "FG_PCT", "EFG_PCT", "POSS_PCT", "SCORE_POSS_PCT", "TOV_POSS_PCT",
  "SF_POSS_PCT", "FT_POSS_PCT", "PLUSONE_POSS_PCT",
  "POSS", "PTS", "FGM", "FGA", "GP", "PERCENTILE"
)

expected_id <- c(
  "PLAY_TYPE", "TYPE_GROUPING", "SEASON_ID",
  "PLAYER_ID", "PLAYER_NAME", "TEAM_ID",
  "TEAM_ABBREVIATION", "TEAM_NAME"
)

FINAL_COLS <- c(
  "SEASON", "SEASON_ID", "ENTITY_TYPE",
  "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
  "PLAY_TYPE", "TYPE_GROUPING",
  "PERCENTILE", "GP", "POSS", "POSS_PCT", "PPP",
  "FG_PCT", "EFG_PCT", "SCORE_POSS_PCT", "TOV_POSS_PCT",
  "SF_POSS_PCT", "FT_POSS_PCT", "PLUSONE_POSS_PCT",
  "PTS", "FGM", "FGA"
)

safe_name <- function(x) {
  gsub("[^A-Za-z0-9_-]", "", x)
}

# ---- single API call helper ----

pull_one <- function(season, pt, side, stype, idx, total) {
  tag <- sprintf("[%03d/%03d] %s | %s | %s | %s",
                 idx, total, season, stype, side, pt)
  message(tag)

  out_file <- file.path(
    CHUNK_DIR,
    sprintf("%s_%s_%s_%s_%s.csv",
            safe_name(PLAYER_OR_TEAM),
            gsub("[^0-9-]", "", season),
            ifelse(stype == "Regular Season", "RS", "PO"),
            substr(side, 1, 3),
            safe_name(pt))
  )

  # reuse local copy if it was already written
  if (file.exists(out_file)) {
    return(vroom::vroom(out_file, show_col_types = FALSE))
  }

  res <- NULL

  for (attempt in 1:4) {
    ok <- TRUE
    try({
      Sys.sleep(runif(1, SLEEP_MIN, SLEEP_MAX))
      res <- hoopR::nba_synergyplaytypes(
        play_type      = pt,
        player_or_team = PLAYER_OR_TEAM,
        season         = season,
        season_type    = stype,
        type_grouping  = side,
        per_mode       = PER_MODE
      )$SynergyPlayType
    }, silent = TRUE)

    if (!is.data.frame(res)) ok <- FALSE
    if (ok) break
    Sys.sleep(2^attempt)
  }

  if (!is.data.frame(res)) {
    warning("Failed (no data): ", tag)
    readr::write_csv(tibble::tibble(), out_file)
    return(tibble::tibble())
  }

  # standardise to upper case
  names(res) <- toupper(names(res))

  # make sure every expected column exists
  for (c in c(expected_num, expected_id)) {
    if (!c %in% names(res)) {
      res[[c]] <- NA_real_
    }
  }

  out <- res %>%
    dplyr::mutate(
      SEASON      = season,
      ENTITY_TYPE = ifelse(PLAYER_OR_TEAM == "P", "Player", "Team")
    ) %>%
    dplyr::mutate(
      dplyr::across(
        dplyr::any_of(expected_num),
        ~ suppressWarnings(as.numeric(.x))
      )
    ) %>%
    dplyr::mutate(
      PLAY_TYPE     = .data[["PLAY_TYPE"]],
      TYPE_GROUPING = .data[["TYPE_GROUPING"]]
    ) %>%
    dplyr::select(dplyr::all_of(FINAL_COLS))

  readr::write_csv(out, out_file)
  out
}

# ---- build grid and run ----

grid <- expand.grid(
  season = SEASONS,
  pt     = PLAYTYPES,
  side   = SIDES,
  stype  = SEASON_TYPES,
  stringsAsFactors = FALSE
)

total <- nrow(grid)

cat("Grid rows:", total, "\n")

# quick smoke test on the first combination
try({
  cat("Running smoke test…\n")
  tmp <- pull_one(grid$season[1], grid$pt[1],
                  grid$side[1], grid$stype[1],
                  1, total)
  cat("Smoke test rows:", nrow(tmp), "\n\n")
}, silent = TRUE)

# full pull
invisible(purrr::pmap(
  cbind(grid, idx = seq_len(total), total = total),
  pull_one
))

# ---- combine chunks and write final CSV ----

files <- list.files(CHUNK_DIR, full.names = TRUE, pattern = "\\.csv$")
cat("Found", length(files), "chunk files in", CHUNK_DIR, "\n")

if (length(files) == 0) {
  stop("No chunk CSV files found in ", CHUNK_DIR)
}

# Read all chunk files into one data frame
synergy <- vroom::vroom(files, show_col_types = FALSE)

# Write the final combined dataset to data/synergy_playtypes_2019_2025_players.csv
readr::write_csv(synergy, FINAL_CSV)

cat("\n===== RESULTS =====\n")
cat("Rows:   ", nrow(synergy), "\n")
cat("Output: ", normalizePath(FINAL_CSV), "\n")
cat("Chunks: ", normalizePath(CHUNK_DIR), "\n")
