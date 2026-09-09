---
oneliner: Tiny dependency-light Levenberg-Marquardt nonlinear least-squares solver for .NET
tags: [nonlinear least squares, levenberg-marquardt, optimization, curve fitting, inverse kinematics, sparse linear algebra, csharp, dotnet, nuget, l-bfgs]
stack: [C#, .NET 8, CSparse]
generated: 2026-09-09
commit: 527657f
placeholder: false
---
A dependency-light .NET port of the Ceres/scipy.optimize.least_squares style solver: dense and
sparse Levenberg-Marquardt with analytic or finite-difference Jacobians, plus an L-BFGS minimizer.
Working v1.0.0, published as a NuGet package with CI release workflow; demos cover IK, curve
fitting, circle fitting and rigid registration. Roadmap tracks box constraints and perf work
(residuals-only trial evaluation, cached sparse symbolic factorization) as v1.1+ items.
