require 'rnn'
require 'optim'
require 'seq2seq'
require 'cutorch'
require 'cunn'
require 'os'


senseVectors = {}
numSenses = 0

getSenseVectors = function()
    sensFile = io.open('senseVectors.csv', 'r') 
    ordFile = io.open('senseOrder.csv', 'r')
    lines = sensFile:read()
    lineo = ordFile:read()
    while lines ~= nil do
        numSenses = numSenses + 1 
        lines = lines:split(",")
        senseVectorSize = #lines
        tmp =torch.Tensor(#lines)
        for i = 1, #lines do
            tmp[i] = tonumber(lines[i])
        end
        senseVectors[lineo] = tmp
        lines = sensFile:read()
        lineo = ordFile:read()
    end
end

getSenseVectors()

cmd = torch.CmdLine()
cmd:option('-model', 'Models/model10')
cmd:option('-set', 'train')
opt = cmd:parse(arg)

batchSize = 2
hiddenSize = 100
t = torch.load("./sent_vectors_train.t7")
xtrain = t['Sent']
ytrain = t['Sense']
loaded_model = t['model']

Secab = loaded_model['Secab']
Index2Secab = loaded_model['Index2Secab']
SecabSize = #Index2Secab


numDocumentsTrain = #xtrain
inputSize = (#xtrain[1])[1]

completed = 0
nextBatch = function(X, Y)
    Inputs = torch.Tensor(batchSize, inputSize)
    Targets = torch.Tensor(batchSize, senseVectorSize)
    for _i = 1, batchSize do
        for _j = 1, inputSize do 
            Inputs[_i][_j] = X[_i + completed][_j]
        end
        tmp = Y[_i + completed]
       
        for _j = 1, senseVectorSize do
            Targets[_i][_j] =  senseVectors[Index2Secab[tmp]][_j]
        end
    end
    Inputs= Inputs:cuda()
    Targets = Targets:cuda()
    completed = completed + batchSize
    return Inputs, Targets
end

feval = function(x_new)
    if x ~= x_new then
        x:copy(x_new)
    end

    dl_dx:zero()

    Inputs, Targets = nextBatch(xtrain, ytrain)

    Output = model:forward(Inputs)

    loss_x = criterion:forward(Output, Targets)

    dloss_doutput = criterion:backward(Output, Targets)

    model:backward(Inputs,dloss_doutput)
    
    model:forget()
    
    return loss_x, dl_dx
end

train = function()
    print(inputSize)
    model = nn.Sequential()
          :add(nn.Linear(inputSize, hiddenSize))
          :add(nn.Tanh())
          :add(nn.Linear(hiddenSize, senseVectorSize))

    criterion = nn.ClassNLLCriterion()

    model:cuda()
    criterion:cuda()

    print('Getting Parameters')
    x, dl_dx = model:getParameters()
    print('Done')

    print('Getting sgd_params')
    sgd_params = {
        learningRate = 1, --changed from 1e-2
        learningRateDecay = 1e-4,
        weightDecay = 0,
        momentum = 0.2    -- changed from 0.2
    }
    print('Done')

    print('Training')
    for j=1, 200 do
        print("------------EPOCH: "..j.."---------------")
        completed = 0

        for i = 1, numDocumentsTrain/batchSize do
            _, fs = optim.sgd(feval, x, sgd_params)
            model:forget()

            if i%100 == 0 then
                print('error for Batch ' .. sgd_params.evalCounter  .. ' is ' .. " # Reviews : ".. sgd_params.evalCounter*batchSize .. " : ".. fs[1])
            end
            
        end

        if j % 5 == 0 then 
            save_model = {}
            save_model['model'] = model
            save_model['Secab'] = Secab
            save_model['Index2Secab'] = Index2Secab
            str_save = 'SentVecModels/' .. j
            torch.save(str_save, save_model)
        end
    end
    print('Done')
end

test = function(modelLoc)
    print(Index2Secab)
    t = torch.load("sent_vectors_test.t7")
    xtest = t['Sent']
    ytest = t['Sense']
    numDocumentsTest = #xtest
    loaded_model_test = torch.load(modelLoc)
    model = loaded_model_test['model']

    print("Testing")
    accCount = 0
    for i = 1, numDocumentsTest do
        testOutput = model:forward(xtest[i])
        max = 0
        argmax = 0
        testOutput = testOutput:double()

        for tsen, tvec in pairs(senseVectors) do
            similarity = torch.abs(torch.dot(testOutput, tvec)/(torch.norm(tvec) * torch.norm(testOutput)))
            if similarity > max then
                max = similarity
                argmax = tsen
            end
            print(similarity)
        end

        pred_sense = argmax
        --print(argmax, Index2Secab[ytest[i]])
        if pred_sense == Index2Secab[ytest[i] ] then
            accCount = accCount + 1
        else
            print(Secab[pred_sense], ytest[i] )
        end
    end
    print(accCount,numDocumentsTest)
end

if opt.set == 'train' then
    train()
else
    test(opt.model)
end
