# Reproduce the dynamic multi-season occupancy model of Birds 2026, 7, 55 (Table 2, Sec 3.6, Sec 3.7).
suppressMessages(library(unmarked))
args <- commandArgs(TRUE); IX <- if (length(args)) args[1] else "NDVI"
OD <- if (length(args) > 1) args[2] else "chrono"
Y  <- as.matrix(read.csv(sprintf("Birds/results/baseline/occ_y_%s_%s.csv", IX, OD)))
SC <- read.csv(sprintf("Birds/results/baseline/occ_sitecovs_%s_%s.csv", IX, OD))
M <- nrow(Y)
# Inter-Hurricanes is the reference level for the detection component (paper Table 2 note)
ord <- if (OD == "chrono") c("Pre-Maria","Inter-Hurricanes","Post-Fiona") else c("Inter-Hurricanes","Post-Fiona","Pre-Maria")
per <- factor(rep(ord, each = 3), levels = c("Inter-Hurricanes","Post-Fiona","Pre-Maria"))
umf <- unmarkedMultFrame(y = Y, numPrimary = 3,
                         siteCovs = data.frame(elev = SC$elev_z, veg = SC$veg_z),
                         obsCovs  = data.frame(period = factor(rep(as.character(per), times = M),
                                                               levels = levels(per))))
fm <- colext(psiformula = ~ elev + veg, gammaformula = ~ veg,
             epsilonformula = ~ elev + veg, pformula = ~ period, data = umf, se = TRUE)
cat(sprintf("\n=== %s / %s  AIC = %.2f  converged=%s ===\n", IX, OD, fm@AIC, fm@opt$convergence == 0))
est <- do.call(rbind, lapply(c("psi","col","ext","det"), function(k) {
  s <- summary(fm[k]); data.frame(component = k, parameter = rownames(s), s, row.names = NULL)}))
names(est) <- c("component","parameter","estimate","SE","z","p")
print(est, digits = 4)
write.csv(est, sprintf("Birds/results/baseline/occupancy_estimates_%s_%s.csv", IX, OD), row.names = FALSE)
# detection probability on the probability scale, by period
b <- coef(fm[ "det" ]); ilog <- function(x) 1/(1+exp(-x))
dp <- data.frame(period = c("Inter-Hurricanes","Post-Fiona","Pre-Maria"),
                 p = ilog(c(b[1], b[1]+b[2], b[1]+b[3])))
print(dp, digits = 4)
write.csv(dp, sprintf("Birds/results/baseline/occupancy_detection_%s_%s.csv", IX, OD), row.names = FALSE)
write.csv(data.frame(index = IX, AIC = fm@AIC, negLogLike = fm@negLogLike, nPars = length(coef(fm)),
                     converged = fm@opt$convergence == 0),
          sprintf("Birds/results/baseline/occupancy_fit_%s_%s.csv", IX, OD), row.names = FALSE)
# predicted local extinction across the elevational gradient at mean standardised NDVI
g <- data.frame(elev = seq(min(SC$elev_z), max(SC$elev_z), length.out = 200), veg = 0)
pe <- predict(fm, type = "ext", newdata = g, appendData = TRUE)
pe$elev_m <- pe$elev * sd(SC$elev_m) + mean(SC$elev_m)
write.csv(pe, sprintf("Birds/results/baseline/occupancy_ext_curve_%s_%s.csv", IX, OD), row.names = FALSE)
# per-site predicted extinction for the spatial map (paper Fig. 6)
ps <- predict(fm, type = "ext", newdata = data.frame(elev = SC$elev_z, veg = SC$veg_z))
write.csv(cbind(SC, ext_pred = ps$Predicted, ext_se = ps$SE),
          sprintf("Birds/results/baseline/occupancy_site_ext_%s_%s.csv", IX, OD), row.names = FALSE)
cat("done\n")
