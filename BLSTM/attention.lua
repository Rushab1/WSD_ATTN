class = require 'class'
argcheck = require 'argcheck'

function tablelength(T)
    local keycount = 0
    keys = {}
    
    for _ in pairs(T) do 
        keycount = keycount + 1
        table.insert(keys, _)
    end
    return keycount, keys
end

cooccurrence = torch.load("cooccurrence.t7")
attnSecabSize, attnSecab = tablelength(cooccurrence)

checkword = "cause"
attnHiddenSize = 100

if Attention == nil then
    Attention = class('Attention')
end

function Attention:score(b, n, input)
        self.w = torch.CudaTensor(b, n):fill(0)
        self.w[{{}, {1}}] = 0
        self.w[{{}, {2}}] = 0
        self.w[{{}, {3}}] = 0
        self.w[{{}, {4}}] = 0
        self.w[{{}, {5}}] = 0
        self.w[{{}, {6}}] = 0
        self.w[{{}, {7}}] = 0
        self.w[{{}, {8}}] = 1
        self.w[{{}, {9}}] = 0
        self.w[{{}, {10}}] = 0
        self.w[{{}, {11}}] = 1

        if self.learning == false then 
            return
        end

        for k = 1, b do
            self.w[{{k}, {1}}] = 0
            self.w[{{k}, {2}}] = 1
            self.w[{{k}, {3}}] = 0
            self.w[{{k}, {4}}] = 0
            self.w[{{k}, {5}}] = 1
            self.w[{{k}, {6}}] = 0
            self.w[{{k}, {7}}] = 0
            self.w[{{k}, {8}}] = 1
            self.w[{{k}, {9}}] = 0
            self.w[{{k}, {10}}] = 0
            self.w[{{k}, {11}}] = 1
            local mse = torch.Tensor(attnSecabSize)
            ru = n
            pos = -1 - math.floor(n/2)

            local sum = 0
            for i = 1, n do
                pos = pos + 1
                for j = 1, attnSecabSize do
                    local tmp1 = 0;
                    
                    local word_count = cooccurrence[attnSecab[j]][pos][ Index2Vocab[ input[i][k] ] ]

                    total_examples = cooccurrence[attnSecab[j]][0][checkword]

                    if word_count ~= nil and word_count/total_examples > 0.05 then
                        tmp1 = tmp1 + 1
                    elseif word_count == nil then
                        word_count = attnSecabSize
                    end
                    
                    mse[j] = word_count/total_examples

                end

                mse = torch.max(nn.Normalize(1):cuda():forward(mse:cuda()))

                self.w[{{k}, {i}}] = mse

                mse = torch.Tensor(attnSecabSize)
            --self.w[{{k}, {}}] = nn.Normalize(1):cuda():forward(self.w[{{k}, {}}])
            end
        end
        self.w_cooc = self.w:clone()
        self.w = self.newm:forward(self.w_cooc)
end

function Attention:__init__(n, b, h, learning , learningRate, learningRateDecay)
    --Input function in the form input_sentence_size * batchSize * hiddenSize (n*b*h)

    if learning == true then 
        self.w_cooc =  torch.CudaTensor(1,n):fill(0)
        self.w = torch.CudaTensor(1,n):fill(0)
        self.learningRate = learningRate
        self.learningRateDecay = learningRateDecay
        self.w = torch.rand(1,n):cuda()
    end

    self.learning = learning

    self.newm = nn.Sequential()
          :add(nn.Linear( n, attnHiddenSize))
          :add(nn.Tanh())
          :add(nn.Linear( attnHiddenSize, n))
          :add(nn.SoftMax())

    self.newm:cuda()

    self.n = n
    self.b = b
    self.h = h 
end

function Attention:forward(inputs, orig_input)

    if self.learning == true then
        self:score(self.b, self.n, orig_input)
        self.output = torch.CudaTensor(self.b, self.h):fill(0)

        for i = 1, self.b do
            self.output[{{i}, {}}] = self.w[{{i},{}}] * inputs[{{}, {i}, {}}]:reshape(self.n, self.h)
        end

        self.input = inputs
        return self.output
    end


    self.output = torch.CudaTensor(self.b, self.h):fill(0)

    self:score(self.b, self.n, orig_input)
    for i = 1, self.b do
        self.output[{{i}, {}}] = self.w[{{i}, {}}] * inputs[{{}, {i}, {}}]:reshape(self.n, self.h)
    end

    self.input = inputs
    return self.output
end

function Attention:backward(output_grad)
    if self.learning == true then
            dloss_dattn = torch.CudaTensor(self.b, self.n)
            
            for j = 1, self.b do
                    dloss_dattn[j] = self.input:select(2,j) * output_grad[{{j}, {}}]:transpose(2,1)
            end

            dloss_dattn = dloss_dattn:cuda()

            attn_op = self.newm:backward(self.w_cooc, dloss_dattn)
            self.newm:updateParameters(self.learningRate)
            self.newm:zeroGradParameters()
    end

    self.input_grad = torch.CudaTensor(self.n, self.b, self.h)
    for i = 1, self.b do
        self.input_grad[{{}, {i}, {}}] = self.w[{{i}, {}}]:transpose(1,2) * output_grad[{{i}, {}}]
    end
    return self.input_grad
end
