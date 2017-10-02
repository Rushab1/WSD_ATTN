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
checkword = "serve"

cooc_score = {}
cooc_score[1] = 10 
cooc_score[2] = 8
cooc_score[3] = 0
cooc_score[4] = 0
cooc_score[5] = 0
cooc_score[6] = 0
cooc_score[7] = 0
cooc_score[8] = 0

if Attention == nil then
    Attention = class('Attention')
end

function Attention:score(b, n, input)
        self.w = torch.CudaTensor(b, n):fill(0)
        self.w[{{}, {1}}] = 0
        self.w[{{}, {2}}] = 1
        self.w[{{}, {3}}] = 0
        self.w[{{}, {4}}] = 0
        self.w[{{}, {5}}] = 1
        self.w[{{}, {6}}] = 0
        self.w[{{}, {7}}] = 0
        self.w[{{}, {8}}] = 1
        self.w[{{}, {9}}] = 0
        self.w[{{}, {10}}] = 0
        self.w[{{}, {11}}] = 1

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
                       --tmp1 = cooccurrence[attnSecab[j]][Index2Vocab[input[i][k]  ]] / cooccurrence[attnSecab[j]][checkword]
                        tmp1 = tmp1 + 1
                    elseif word_count == nil then
                        word_count = attnSecabSize
                    end
                    
                    --mse = mse + tmp1
                    mse[j] = word_count/total_examples

                end

                --print("____________________")
                --print(mse:resize(1,attnSecabSize))
                mse = torch.max(nn.Normalize(1):cuda():forward(mse:cuda()))

                --mse = attnSecabSize - mse

                if self.w[{{k}, {i}}][1][1] ~= 0 then
                    --local tmp = i + 2
                    local tmp = i
                    if tmp > n then
                        tmp = n
                    end
                    
                    sum = sum + mse
                    self.w[{{k}, {tmp}}] = sum
                    
                    --mse = 0
                    sum = 0
                    mse = torch.Tensor(attnSecabSize)
                else
                    sum = sum + mse
                    mse = torch.Tensor(attnSecabSize)
                end

            end

            self.w[{{k}, {}}] = nn.Normalize(1):cuda():forward(self.w[{{k}, {}}])
            for i = 1, n do
                if self.w[{{k}, {i}}][1][1]  < 1/attnSecabSize then
                    self.w[{{k}, {i}}] = 0
                end
            end
            self.w[{{k}, {}}] = nn.Normalize(1):cuda():forward(self.w[{{k}, {}}])
        end
        
end

function Attention:__init__(n, b, h, learning , learning_rate)
    --Input function in the form input_sentence_size * batchSize * hiddenSize (n*b*h)

    if learning == true then 
        self.w = torch.CudaTensor(1,n):fill(0)
        self.w[{{}, {n}}] = 1
        self.w[{{}, {8}}] = 1
        self.learning_rate = learning_rate
        self.w = torch.rand(1,n):cuda()
    end

    self.learning = learning

    newm = nn.Sequential()
          :add(nn.Linear(h * n, h))

    newm:cuda()

    self.n = n
    self.b = b
    self.h = h 
end

function Attention:forward(inputs, orig_input)

    if self.learning == true then
        self.output = torch.CudaTensor(self.b, self.h):fill(0)

        for i = 1, self.b do
            self.output[{{i}, {}}] = self.w * inputs[{{}, {i}, {}}]:reshape(self.n, self.h)
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
            tmp = torch.Tensor(self.b, self.h * self.n)
            
            for j = 1, self.b do
                for k = 1, self.n do
                    tmp[{{j}, {self.h*(k -1) + 1, self.h * k }}][1] = torch.Tensor(self.h):copy(self.input[{{k}, {j}}][1][1])
                end
            end

            for k = 1, self.n do
                newm.modules[1].weight[{{}, {self.h * (k-1) + 1, self.h * k}}] = self.w[1][k]
            end

            tmp = tmp:cuda()
            newm:zeroGradParameters()
            attn_op = newm:backward(tmp, output_grad)
            newm:updateParameters(self.learning_rate)

            for k = 1, self.n do
                self.w[1][k] = torch.mean(newm.modules[1].weight[{{}, {self.h * (k-1) + 1, self.h * k}}])
            end
    end

    self.input_grad = torch.CudaTensor(self.n, self.b, self.h)
    for i = 1, self.b do
        if self.learning == true then
            self.input_grad[{{}, {i}, {}}] = self.w:transpose(1,2) * output_grad[{{i}, {}}]
        else
            self.input_grad[{{}, {i}, {}}] = self.w[{{i}, {}}]:transpose(1,2) * output_grad[{{i}, {}}]
        end
    end
    return self.input_grad
end
