
using GraphvizDotLang: digraph, edge, node, save, attr

g = digraph() |>
  attr(:node, shape = "record") |>
  node("n1", label = "{ name: a | data: 0 | grad: 0.0 }") |>
  node("n2", label = "{ name: b | data: 1 | grad: 0.0 }") |>
  edge("n1", "n2")

save(g, "test.png", format = "png")

