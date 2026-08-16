#!/usr/bin/env Rscript
# Causal survival forest bridge: reads a dataset CSV, writes population
# benefit estimates S1(h)-S0(h) with SEs at the requested horizons.
# Usage: Rscript csf_bridge.R <in.csv> <out.csv> <horizon1,horizon2,...> [num.trees]
suppressMessages(library(grf))
args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(args[1])
horizons <- as.numeric(strsplit(args[2 + 1], ",")[[1]])
num_trees <- if (length(args) >= 4) as.integer(args[4]) else 1000
out <- args[2]

Xcols <- grep("^X", names(dat), value = TRUE)
X <- as.matrix(dat[, Xcols])
res <- data.frame()
for (h in horizons) {
  cs <- causal_survival_forest(X, dat$Ttil, dat$A, dat$Delta,
                               target = "survival.probability", horizon = h,
                               num.trees = num_trees, seed = 1)
  ate <- average_treatment_effect(cs)
  res <- rbind(res, data.frame(horizon = h, est = ate[["estimate"]],
                               se = ate[["std.err"]]))
}
write.csv(res, out, row.names = FALSE)
