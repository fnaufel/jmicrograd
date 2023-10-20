module Jmicrograd
export Value, newValue, print, clear_grads!, backward, relu
export @nv
export build_dot

import Base.:*
import Base.:+
import Base.:-
import Base.:/
import Base.:^
import Base.print

using GraphvizDotLang: digraph, edge, node, save, attr

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
function build_dot(v::Value)
  
  # Build list of nodes
  global topo = []
  global visited = Set()
  build_topo(v)
  # reverse!(topo)
  
  # Create dictionary: node => index in the list
  d = IdDict([(node, i) for (node, i) = zip(topo, 1:length(topo))])

  # Initialize graph
  g = digraph() |> attr(:node, shape = "record")

  # Enter nodes in the graph
  for this_node in topo
    this_id   = d[this_node]
    this_name = this_node.name
    this_data = this_node.data
    this_grad = this_node.grad
    g = g |> node(
      "node$this_id",
      label = "{ $this_name | data: $this_data | grad: $this_grad }"
    )
  end

  # Enter edges in the graph
  for this_node in topo
    this_id   = d[this_node]
    for child in this_node.prev
      child_id = d[child]
      g = g |> edge("node$this_id", "node$child_id")
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
  a + newValue(b)
end

function +(a::Number, b::Value)
  newValue(a) + b
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
  a * newValue(b)
end

function +(a::Number, b::Value)
  newValue(a) * b
end


#######
# Sub #
#######
function -(a::Value, b::Value)
  a + (-1)*b
end

function -(a::Value, b::Number)
  a - newValue(b)
end

function -(a::Number, b::Value)
  newValue(a) - b
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
function ^(a::Value, b::Number)
  out = newValue(a.data^b, [a]; name = "^$b")
  out._backward = function()
    a.grad += b * a.data^(b - 1) * out.grad
  end
  out
end


############
# Division #
############
function /(a::Value, b::Value)
  a * b^1
end

function /(a::Value, b::Number)
  a / newValue(b)
end

function /(a::Number, b::Value)
  newValue(a) / b
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
  relu(newValue(a))
end

end # module

