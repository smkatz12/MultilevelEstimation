# Sets up problem structures
using GridInterpolations

mutable struct GriddedProblem
    grid_points::Dict # Dictionary of grid cutpoints
    grid::RectangleGrid # Grid of points
    sim::Function # Function that takes in parameter set and runs n sims
    pfail_threshold::Float64 # probability of failure threshold for determining if safe
    conf_threshold::Float64 # desired confidence
    is_safe::Vector # Vector of booleans (matching grid indices)
end