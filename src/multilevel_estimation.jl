# Sets up problem structures
using GridInterpolations

mutable struct GriddedProblem
    grid::RectangleGrid # Grid of points
    sim::Callable # Function that takes in parameter set and runs n sims
    pfail_threshold::Float64 # probability of failure threshold for determining if safe
    is_safe::Vector # Vector of booleans (matching grid indices)
end