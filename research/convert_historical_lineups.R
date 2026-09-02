#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: convert_historical_lineups.R INPUT.rds OUTPUT.csv.gz")
}

if (!requireNamespace("data.table", quietly = TRUE)) {
  stop("data.table is required")
}

input <- args[[1]]
output <- args[[2]]
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
data.table::fwrite(readRDS(input), output, compress = "gzip")
