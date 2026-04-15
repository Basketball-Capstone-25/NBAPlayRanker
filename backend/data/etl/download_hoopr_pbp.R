# Download NBA play-by-play data from hoopR and save as parquet
# for the Python ETL pipeline.
#
# Usage:  Rscript backend/data/etl/download_hoopr_pbp.R
#
# Requires: hoopR, arrow, progressr, tictoc (install via install.packages)

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(hoopR, arrow, progressr, tictoc)

output_path <- file.path("backend", "data", "pbp", "nba_pbp_2021_present.parquet")

cat("Downloading NBA PBP data (2021–present) via hoopR...\n")
tictoc::tic()
progressr::with_progress({
  nba_pbp <- hoopR::load_nba_pbp(2021:hoopR::most_recent_nba_season())
})
tictoc::toc()

cat(sprintf("Downloaded %s rows x %s columns\n", format(nrow(nba_pbp), big.mark = ","), ncol(nba_pbp)))
cat(sprintf("Columns: %s\n", paste(names(nba_pbp), collapse = ", ")))

# Save as parquet for the Python pipeline
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
arrow::write_parquet(nba_pbp, output_path)
cat(sprintf("Saved to: %s (%.1f MB)\n", output_path, file.size(output_path) / 1e6))
