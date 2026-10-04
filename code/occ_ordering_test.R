# Does the observation-covariate stacking order change the published detection component?
# Everything is held fixed at the authors' own values -- their detection matrix, their site
# covariates, their formulas, their optimiser settings -- and only `each=` vs `times=` changes.
suppressMessages(library(unmarked))
Y  <- as.matrix(read.csv("Birds/results/baseline/occ_y_NDVI_author.csv"))
SC <- read.csv("Birds/results/baseline/occ_sitecovs_NDVI_author.csv")
M <- nrow(Y)
per <- rep(c("Pre-Maria","Inter-Hurricanes","Post-Fiona"), each = 3)   # column -> period
lev <- c("Inter-Hurricanes","Post-Fiona","Pre-Maria")                  # reference = Inter-Hurricanes
sc  <- data.frame(Elevation = SC$elev_z, NDVI = SC$veg_z)
fit <- function(mode) {
  oc <- if (mode == "as_published")            # rep(..., each = nrow(Y)) : observation-major
          data.frame(Period = factor(rep(per, each = M), levels = lev))
        else                                    # rep(..., times = nrow(Y)): site-major, what
          data.frame(Period = factor(rep(per, times = M), levels = lev))   # unmarked expects
  umf <- unmarkedMultFrame(y = Y, numPrimary = 3, siteCovs = sc, obsCovs = oc)
  colext(~ Elevation + NDVI, ~ NDVI, ~ Elevation + NDVI, ~ Period, data = umf,
         method = "BFGS", se = TRUE, control = list(maxit = 10000, trace = 0))
}
res <- list()
for (mode in c("as_published", "site_major")) {
  fm <- fit(mode)
  e <- do.call(rbind, lapply(c("psi","col","ext","det"), function(k) {
        s <- summary(fm[k]); data.frame(component = k, parameter = rownames(s), s, row.names = NULL)}))
  names(e) <- c("component","parameter","estimate","SE","z","p")
  e$mode <- mode; e$AIC <- fm@AIC; e$converged <- fm@opt$convergence == 0
  res[[mode]] <- e
  cat(sprintf("\n=== %s : AIC = %.4f  converged = %s ===\n", mode, fm@AIC, fm@opt$convergence == 0))
  print(e[, c("component","parameter","estimate","SE","z","p")], digits = 4)
}
out <- do.call(rbind, res)
write.csv(out, "Birds/results/baseline/occupancy_ordering_test.csv", row.names = FALSE)
# detection on the probability scale under each stacking order
for (mode in names(res)) {
  b <- res[[mode]][res[[mode]]$component == "det", "estimate"]
  il <- function(x) 1/(1+exp(-x))
  cat(sprintf("%s detection p: Inter-Hurricanes=%.4f Post-Fiona=%.4f Pre-Maria=%.4f\n",
              mode, il(b[1]), il(b[1]+b[2]), il(b[1]+b[3])))
}
