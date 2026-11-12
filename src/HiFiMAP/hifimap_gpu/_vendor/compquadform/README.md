# Davies' algorithm (qfc), vendored from CompQuadForm 1.4.4

`qfc_original.cpp` is an unmodified copy of CompQuadForm's `src/qfc.cpp`
(Davies, 1980, "The Distribution of a Linear Combination of Chi-Squared
Random Variables", Algorithm AS 155), downloaded from
https://cran.r-project.org/src/contrib/CompQuadForm_1.4.4.tar.gz. It is the
exact C++ code R's `CompQuadForm::davies()` calls via `.C("qfc", ...)`.
CompQuadForm is licensed GPL (>= 2).

`qfc_standalone.cpp` is the same file with only its two `#include`
lines changed (`<R.h>` and `"Rmath.h"` to `<cmath>` and `<cstdlib>`).
No other line differs — the file includes those headers but calls no R API
function from them (verified by inspection: no `error()`, `Rprintf`,
`qchisq`, or other R/Rmath symbols appear anywhere in the algorithm). The
goal is to call the identical algorithm from Python, not to reimplement it.
