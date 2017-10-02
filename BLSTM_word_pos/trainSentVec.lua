require 'rnn'
require 'optim'
require 'seq2seq'
require 'cutorch'
require 'cunn'

cmd = torch.CmdLine()
cmd:option("-set", "train")
opt = cmd:parse(arg)

batchSize = 50
t = torch.load("./sent_vectors_train.t7")
xtrain = t['Sent']
ytrain = t['Sense']
loaded_model = t['model']

Secab = loaded_model['Secab']
Index2Secab = loaded_model['Index2Secab']
SecabSize = #Index2Secab

t = torch.load("./sent_vectors_test.t7")
xtest = t['Sent']
ytest = t['Sense']

numDocumentsTrain = #xtrain
numDocumentsTest = #xtest
inputSize = (#xtrain[1])[1]

completed = 0
nextBatch = function(X, Y)
    Inputs = torch.Tensor(batchSize, inputSize)
    Targets = torch.IntTensor(batchSize)
    for _i = 1, batchSize do
        for _j = 1, inputSize do 
            Inputs[_i][_j] = X[_i + completed][_j]
        end
        Targets[_i] = Secab[Y[_i + completed]]
    end
    --print(Inputs)
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
    --print("T1")

    Inputs, Targets = nextBatch(xtrain, ytrain)
    --print("T2")

    Output = model:forward(Inputs)
    --print("T3")

    loss_x = criterion:forward(Output, Targets)
    --print("T4")

    dloss_doutput = criterion:backward(Output, Targets)
    --print("T5")

    b = model:backward(Inputs,dloss_doutput)
    --print("T6")

    --pr, wd = Output[1]:topk((#Output)[2], 1, true, true)
    --print("---------------")
    --print(Output[1])
    --print(wd[1], Inputs[1][1])


    model:forget()
    return loss_x, dl_dx
end

train = function()
    model = nn.Sequential()
          :add(nn.Linear(inputSize, SecabSize))
          :add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()

    model:cuda()
    criterion:cuda()

    print('Getting Parameters')
    x, dl_dx = model:getParameters()
    print('Done')

    print('Getting sgd_params')
    sgd_params = {
        learningRate = 0.1, --changed from 1e-2
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

            if i%10 == 0 then
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
    loaded_model_test = torch.load(modelLoc)
    model = loaded_model_test['model']

    print("Testing")
    accCount = 0
    for i = 1, numDocumentsTest do
        probabilities = model:forward(xtest[i])
        probabilities, wordIds = probabilities:topk((#probabilities)[1], 1, true, true)
        pred_sense = wordIds[1]
        if pred_sense == ytest[i] + 1 then
            accCount = accCount + 1
        else
            print(pred_sense, ytest[i] + 1 )
        end
    end
    print(accCount,numDocumentsTest)
end

if opt.set == 'train' then
    train()
else
    test()
end
