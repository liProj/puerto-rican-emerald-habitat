# Which layout does unmarked expect for obsCovs? Label each cell uniquely as "site<i>_occ<j>"
# and read back what unmarked stores, so the mapping is observed rather than assumed.
suppressMessages(library(unmarked))
M <- 4; J <- 6                         # 4 sites, 3 primary x 2 secondary
y <- matrix(0:(M*J-1), nrow = M, byrow = TRUE)   # cell value = its own index, row-major
lab <- outer(1:M, 1:J, function(i, j) paste0("s", i, "_o", j))
umf_site <- unmarkedMultFrame(y = y, numPrimary = 3,
              obsCovs = data.frame(tag = as.vector(t(lab))))   # site-major: t() then vec
umf_obs  <- unmarkedMultFrame(y = y, numPrimary = 3,
              obsCovs = data.frame(tag = as.vector(lab)))      # observation-major: column-major
cat("y (row = site, col = occasion):\n"); print(y)
cat("\n-- site-major input: first 12 stored obsCovs rows --\n")
print(head(obsCovs(umf_site), 12))
cat("\n-- observation-major input: first 12 stored obsCovs rows --\n")
print(head(obsCovs(umf_obs), 12))
cat("\nunmarked stores obsCovs as one block of J rows per site, in site order.\n")
cat("So row k corresponds to site ceiling(k/J), occasion ((k-1) mod J)+1.\n")
cat("Correct input is therefore as.vector(t(cell_matrix)) = rep(per_occasion, times = M).\n")
