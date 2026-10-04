# Which construction reproduces the paper's AIC = 2878.81 and Table 2?
suppressMessages(library(unmarked))
tryfit <- function(IX, OD, mode) {
  Y  <- as.matrix(read.csv(sprintf("Birds/results/baseline/occ_y_%s_%s.csv", IX, OD)))
  SC <- read.csv(sprintf("Birds/results/baseline/occ_sitecovs_%s_%s.csv", IX, OD))
  M <- nrow(Y)
  ord <- if (OD == "chrono") c("Pre-Maria","Inter-Hurricanes","Post-Fiona") else c("Inter-Hurricanes","Post-Fiona","Pre-Maria")
  lev <- c("Inter-Hurricanes","Post-Fiona","Pre-Maria")
  sc <- data.frame(elev = SC$elev_z, veg = SC$veg_z)
  a <- list(y = Y, numPrimary = 3, siteCovs = sc)
  if (mode == "site-major")  a$obsCovs <- data.frame(period = factor(rep(rep(ord, each = 3), times = M), levels = lev))
  if (mode == "obs-major")   a$obsCovs <- data.frame(period = factor(rep(rep(ord, each = 3), each  = M), levels = lev))
  if (mode == "yearlySite")  a$yearlySiteCovs <- data.frame(period = factor(rep(ord, times = M), levels = lev))
  umf <- do.call(unmarkedMultFrame, a)
  fm <- try(colext(~ elev + veg, ~ veg, ~ elev + veg, ~ period, data = umf, se = TRUE), silent = TRUE)
  if (inherits(fm, "try-error")) return(NULL)
  b <- coef(fm); il <- function(x) 1/(1+exp(-x)); d <- coef(fm["det"])
  cat(sprintf("%-6s %-7s %-11s AIC=%9.2f  det:int=%7.3f PF=%7.3f PM=%7.3f  p=(%.3f,%.3f,%.3f)\n",
      IX, OD, mode, fm@AIC, d[1], d[2], d[3], il(d[1]), il(d[1]+d[2]), il(d[1]+d[3])))
  invisible(fm)
}
for (od in c("chrono","alpha")) for (md in c("site-major","obs-major","yearlySite")) tryfit("NDVI", od, md)
cat("\npaper:                          AIC=  2878.81  det:int= -0.736 PF=  0.120 PM= -0.566  p=(0.324,0.351,0.214)\n")
