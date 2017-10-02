class = require 'class'
argcheck = require 'argcheck'

windowSize = 11

if Attention == nil then
    Attention = class('Attention')
end

function Attention:score(b, n)
    if self.w == nil then
        self.w = torch.CudaTensor(b, n):fill(0)
        self.w[{{}, {n}}] = 1
        self.w[{{}, {8}}] = 1
    end
end

function Attention:__init__(n, b, h)
    --Input function in the form input_sentence_size * batchSize * hiddenSize (n*b*h)
    self.n = n
    self.b = b
    self.h = h 
end

function Attention:forward(inputs)
    self.output = torch.CudaTensor(self.b, self.h):fill(0)

    self:score(self.b, self.n)
    for i = 1, self.b do
        self.output[{{i}, {}}] = self.w[{{i}, {}}] * inputs[{{}, {i}, {}}]:reshape(self.n, self.h)
    end

    self.input = inputs
    return self.output
end

function Attention:backward(output_grad)
    self.input_grad = torch.CudaTensor(self.n, self.b, self.h)

    for i = 1, self.b do
        self.input_grad[{{}, {i}, {}}] = self.w[{{i}, {}}]:transpose(1,2) * output_grad[{{i}, {}}]
    end
    return self.input_grad
end
