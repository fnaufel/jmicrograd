module Jmicrograd
export Value, newValue, print, clear_grads!, backward, relu
export @nv
export build_dot

import Base.:*
import Base.:+
import Base.:-
import Base.:/
import Base.:^
import Base.:exp
import Base.:inv
import Base.:literal_pow
import Base.print

using GraphvizDotLang: digraph, edge, node, save, attr, HTML
using Printf: @sprintf


#########
# Class #
#########
mutable struct Value
  data::Float64
  prev::Vector{Value}
  grad::Float64
  _backward
  name::String
end


###############
# Constructor #
###############
function newValue(
  data = 0,
  prev = [],
  grad = 0,
  _backward = ()->nothing;
  name = ""
)
  Value(data, prev, grad, _backward, name)
end


##########################################################
# Macro to construct instance with same name as variable #
##########################################################
macro nv(name, data)
  thisname = string(name)
  return quote
    $(esc(name)) = newValue($data; name = $thisname)
  end
end


#########
# Print #
#########
function print(v::Value)
  print(print_helper(v))
end

function print_helper(v::Value, indent = "")
  # This node's fields
  s =
    indent * " |\\------------\n" *
    indent * " |  name: " * v.name * "\n" *
    indent * " |  data: " * string(v.data) * "\n" *
    indent * " |  grad: " * string(v.grad) * "\n"
  # Previous nodes
  s2 = ""
  if length(v.prev) > 0 
    s = s * indent * " |  prev " * "\n"
    indent = indent * " | "
    s2 =
      s2 *
      mapreduce(x -> print_helper(x, indent), string, v.prev, init = "")
  end
  # Return this node's fields and ancestors
  s * s2
end


################################
# Clear gradients in this node #
################################
function clear_grads!(v::Value)
  v.grad = 0
  for vv in v.prev
    clear_grads!(vv)
  end
end


##############################
# Globals used by build_topo #
##############################
topo = []
visited = Set()


############################################################################
# Topologically sort the expression graph, storing the list in global topo #
############################################################################
function build_topo(v::Value)
  if !(v in visited)
    push!(visited, v)
    for child in v.prev
      build_topo(child)
    end
    push!(topo, v)
  end
end


#########################
# Build graph using dot #
#########################
# Traverse the list once, generating the nodes
# Traverse it again, generating edges
# Return digraph object
function build_dot(
  v::Value,
  title::String = "";
  rankdir = "LR",
  ranksep = 1,
  nodesep = 0.5,
  fontsize = 24,
  dpi = 150,
  digits = 3
)
  
  # Build list of nodes
  global topo = []
  global visited = Set()
  build_topo(v)
  
  # Create dictionary: node => index in the list
  d = IdDict([(node, i) for (node, i) = zip(topo, 1:length(topo))])

  # Pad label
  if title != ""
    title *= "\n\n"
  end
  
  # Initialize graph
  g = digraph() |>
    attr(:graph, rankdir = rankdir) |>
    attr(:graph, ranksep = string(ranksep)) |>
    attr(:graph, nodesep = string(nodesep)) |>
    attr(:graph, fontsize = string(fontsize)) |>
    attr(:graph, dpi = string(dpi)) |>
    attr(:graph, label = title) |>
    attr(:graph, labelloc = "t") |>
    attr(:node, shape = "none") |>
    attr(:node, margin = "0") |>
    attr(:node, fixedsize = "false")

  # Enter nodes in the graph
  for this_node in topo
    this_id   = d[this_node]
    this_name = this_node.name
    this_data = round(this_node.data, digits = digits)
    this_grad = round(this_node.grad, digits = digits)
    g = g |> node(
      "node$this_id",
      label = HTML(
        @sprintf "
        <table cellborder=\"1\" border=\"0\" cellspacing=\"0\" cellpadding=\"5\">
        <tr><td bgcolor=\"gray\"><b>%s</b></td></tr>
        <tr><td align=\"right\"><font color=\"blue\">%s »</font></td></tr>
        <tr><td align=\"left\"><font color=\"red\">« %s</font></td></tr>
        </table>" this_name this_data this_grad
      )
    )
  end

  # Enter edges in the graph
  for this_node in topo
    this_id   = d[this_node]
    for child in this_node.prev
      child_id = d[child]
      g = g |> edge("node$child_id", "node$this_id")
    end
  end

  g

end


####################################
# Backward pass starting at node v #
####################################
function backward(v::Value)
  clear_grads!(v)
  # topological order all of the children in the graph
  global topo = []
  global visited = Set()
  build_topo(v)
  v.grad = 1
  for node in reverse(topo)
    node._backward()
  end
end


#######
# Add #
#######
function +(a::Value, b::Value)
  out = newValue(a.data + b.data, [a, b]; name = "+")
  out._backward = function()
    a.grad += out.grad
    b.grad += out.grad
  end
  out
end

function +(a::Value, b::Number)
  a + newValue(b, name = string(b))
end

function +(a::Number, b::Value)
  newValue(a, name = string(a)) + b
end


########
# Mult #
########
function *(a::Value, b::Value)
  out = newValue(a.data * b.data, [a, b]; name = "*")
  out._backward = function()
    a.grad += b.data * out.grad
    b.grad += a.data * out.grad
  end
  out
end

function *(a::Value, b::Number)
  a * newValue(b, name = string(b))
end

function +(a::Number, b::Value)
  newValue(a, name = string(a)) * b
end


#######
# Sub #
#######
function -(a::Value, b::Value)
  a + (-1)*b
end

function -(a::Value, b::Number)
  a - newValue(b, name = string(b))
end

function -(a::Number, b::Value)
  newValue(a, name = string(a)) - b
end


############
# Negation #
############
function -(a::Value)
  -1 * a
end


#########
# Power #
#########
# From the docs:
#
# If y is an Int literal (e.g. 2 in x^2 or -3 in x^-3), the Julia code
# x^y is transformed by the compiler to Base.literal_pow(^, x,
# Val(y)), to enable compile-time specialization on the value of the
# expo- nent. (As a default fallback we have Base.literal_pow(^, x,
# Val(y)) = ^(x,y), where usually ^ == Base.^ unless ^ has been
# defined in the calling namespace.) If y is a negative integer
# literal, then Base.literal_pow transforms the operation to inv(x)^-y
# by default, where -y is positive.
function ^(a::Value, b::Number)
  out = newValue(a.data^b, [a]; name = "^$b")
  out._backward = function()
    a.grad += b * a.data^(b - 1) * out.grad
  end
  out
end


#######
# inv #
#######
function inv(a::Value)
  out = newValue(1/(a.data), [a]; name = "1/x")
  out._backward = function()
    a.grad += -1/(a.data^2) * out.grad
  end
  out
end


############
# Division #
############
function /(a::Value, b::Value)
  a * b^-1
end

function /(a::Value, b::Number)
  a / newValue(b, name = string(b))
end

function /(a::Number, b::Value)
  newValue(a, name = string(a)) / b
end


########
# ReLU #
########
function relu(a::Value)
  out = newValue(max(0, a.data), [a]; name = "ReLU")
  out._backward = function()
    a.grad += (a.data <= 0 ? 0 : 1) * out.grad
  end
  out
end  

function relu(a::Number)
  relu(newValue(a, name = string(a)))
end


#######
# exp #
#######
function exp(a::Value)
  out = newValue(exp(a.data), [a]; name = "exp")
  out._backward = function()
    a.grad += out.data * out.grad
  end
  out
end


end # module

