import Base.:*
import Base.:+
import Base.:-
import Base.:/
import Base.print


# Class
mutable struct Value
  data::Float64
  op::String
  prev::Vector{Value}
  grad::Float64
  _backward
end


# Constructor
function newValue(data = 0, op = "", prev = [], grad = 0, _backward = nothing)
  Value(data, op, prev, grad, _backward)
end


# Print
function print(v::Value)
  print(print_helper(v))
end

function print_helper(v::Value, indent = "")
  # This node's fields
  s =
    indent * " |\\------------\n" *
    indent * " |  data: " * string(v.data) * "\n" *
    indent * " |  grad: " * string(v.grad) * "\n" * 
    indent * " |  op:   " * v.op * "\n"
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


# Clear gradients in this node and its previous nodes
function clear_grads(v::Value)
  v.grad = 0
  for vv in v.prev
    clear_grads(vv)
  end
end


# Add
function +(a::Value, b::Value)
  out = newValue(a.data + b.data, "+", [a, b])
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


# Mult
function *(a::Value, b::Value)
  out = newValue(a.data * b.data, "*", [a, b])
  out._backward = function()
    a.grad += b.grad * out.grad
    b.grad += a.grad * out.grad
  end
  out
end

function *(a::Value, b::Number)
  a * newValue(b)
end

function +(a::Number, b::Value)
  newValue(a) * b
end


# Sub
function -(a::Value, b::Value)
  a + (-1)*b
end

function -(a::Value, b::Number)
  a - newValue(b)
end

function -(a::Number, b::Value)
  newValue(a) - b
end


# Negation
function -(a::Value)
  -1 * a
end


# Power
function ^(a::Value, b::Number)
  out = newValue(a.data^b, "^"*b, [a])
  out._backward = function()
    a.grad += b * a.data^(b - 1) * out.grad
  end
  out
end


# Division
function /(a::Value, b::Value)
  a * b^1
end

function /(a::Value, b::Number)
  a / newValue(b)
end

function /(a::Number, b::Value)
  newValue(a) / b
end


# ReLU
function relu(a::Value)
  out = newValue(max(0, a.data), "ReLU", [a])
  out._backward = function()
    a.grad += b * (a.data <= 0 ? 0 : 1) * out.grad
  end
  out
end  

function relu(a::Number)
  relu(newValue(a))
end
