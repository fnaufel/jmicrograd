include("jmicrograd.jl")
using .Jmicrograd
using GraphvizDotLang


#######################################
# Addition with 3 values and backward #
#######################################
println("\n 1 + 2 + 3:")
@nv v1 1
@nv v2 2
@nv v3 3
soma = v1 + v2 + v3
backward(soma)
print(soma)


######################################################
# Addition and multiplication in the same expression #
######################################################
println("\n 1 + 2 * 3:")
@nv v1 1
@nv v2 2
@nv v3 3
e = v1 + v2 * v3
backward(e)
print(e)


########
# ReLU #
########
println("\n 1 + 2 * 3:")
@nv v1 1
@nv v2 2
@nv v3 3
e = relu(v1 + v2 * v3)
backward(e)
print(e)


#############
# DOT graph #
#############
g = build_dot(e, "ReLU(v1 + v2 * v3)")
println("\nDOT graph:\n")
println(g)
GraphvizDotLang.save(g, "example1.png", format = "png")
println("Saved to example1.png")


##############################
# DOT graph: parallel arrows #
##############################
e = relu(v1 + v2 * v2)
backward(e)
g = build_dot(e)
println("\nDOT graph:\n")
println(g)
GraphvizDotLang.save(g, "example2.png", format = "png")
println("Saved to example2.png")


##############################
# DOT graph: parallel arrows #
##############################
e = relu(v1 + v2^2)
backward(e)
g = build_dot(e)
println("\nDOT graph:\n")
println(g)
GraphvizDotLang.save(g, "example3.png", format = "png")
println("Saved to example3.png")


