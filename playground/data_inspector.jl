
using DataFrames: DataFrame, select
using DelimitedFiles: readdlm
using Plots: plot
using CSV: File
using StatsPlots: @df

relative_csv_path = "./../src/data/raw/heat_transfer_finn.csv"
df = DataFrame(File(relative_csv_path))

#plot a data frame in julia using StatsPlots



@df df plot([:T1],:G, label = "T1")
@df df plot!([:T2],:G, label = "T2")
@df df plot!([:T3],:G, label = "T3")
@df df plot!([:T4],:G, label = "T4")
@df df plot!([:T5],:G, label = "T5")
@df df plot!([:T6],:G, label = "T6")
@df df plot!([:T7],:G, label = "T7")
@df df plot!([:T8],:G, label = "T8")
@df df plot!([:T9],:G, label = "T9", xlabel = "T [K]", ylabel = "G")
