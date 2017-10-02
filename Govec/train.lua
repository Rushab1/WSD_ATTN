require 'rnn'
require 'optim'
--require 'seq2seq'
require 'cutorch'
require 'cunn'
if w2v == nil then 
    w2v =  require 'w2vutils.lua'
end
---------------------------------

cutorch.setDevice(1)
torch.setdefaulttensortype('torch.CudaTensor')
print('Running with CUDA on GPU')
----------------------------------

file_txt=io.open('text_words.csv','r')
file_smy=io.open('summary_words.csv','r')

batchSize = 5
numDocuments = 20000
hiddenSize = 300
minVocabCount = 0
print('Begin')

--*************************************************************

reviews={}
summaries={}

rf = io.open('text_words.csv','r')
sf = io.open('summary_words.csv','r')

rev = rf:read():lower()
while rev ~= nil do
    table.insert(reviews, rev:lower())
    rev = rf:read()
end

sum = sf:read():lower()
while sum ~= nil do
    table.insert(summaries, sum:lower())
    sum = sf:read()
end

revcnt = 1
sumcnt = 1 

function nextBatch()
    --print("ENTERING nextBatch")
    --local encoderInputs, decoderInputs, decoderTargets = {}, {}, {}

    sentBatch_text = {}
    sentBatch_smy={}
    local max_len_text = 0
    local max_len_smy = 0 
    local j=0, j2

    for j=1,batchSize do 
        local sent_txt = {}
        local sent_smy = {}

        line = reviews[revcnt]; revcnt = revcnt + 1
        while line ~= "" and line~=nil do 
            table.insert(sent_txt,line)
            --line = file_txt:read()
            line = reviews[revcnt]; revcnt = revcnt + 1
            if line ~= nil then
                line = line:lower()
            end
        end
        table.insert(sentBatch_text,sent_txt)
        sent_txt_Inputs =sent_txt 
        
        if max_len_text < #sent_txt then
            max_len_text = #sent_txt
        end

        table.insert(sent_smy, '<go>')
        line = summaries[sumcnt]; sumcnt = sumcnt + 1
        while line ~= "" and line~= nil  do     
            table.insert(sent_smy, line)
            line = summaries[sumcnt]; sumcnt = sumcnt + 1
            if line~=nil then
                line = line:lower()
            end
        end 
    
        --print('======================================')
        --print(sent_txt,sent_smy)
        --print('======================================')
        table.insert(sent_smy, '<eos>')
        table.insert(sentBatch_smy,sent_smy)
        
        if max_len_smy < #sent_smy then
            max_len_smy = #sent_smy
        end 
    end 

     encoderInputs = torch.DoubleTensor(max_len_text, batchSize, 300):fill(0)
    --encoderInputs = {}
    --encoderInputs = torch.IntTensor(max_len_text,batchSize):fill(0)
    local decoderInputs = torch.IntTensor(max_len_smy-1,batchSize):fill(0)
    local decoderTargets= torch.IntTensor(max_len_smy-1,batchSize):fill(0)

    for j=1,batchSize do
        eosOffset = max_len_text - #sentBatch_text[j] --left paddding

        for j2=1,#sentBatch_text[j] do   
                if w2v.w2vvocab[sentBatch_text[j][j2]] ~=nil then
                    encoderInputs[j2+eosOffset][j] = w2v.M[w2v.w2vvocab[sentBatch_text[j][j2]]]
                end
            --if  Vocab[sentBatch_text[j][j2]] ~= nil then 
                --print("NB1",sentBatch_text[j][j2],j,j2)
                --_enc = encoderInputs
                --encoderInputs[j2+eosOffset][j] = Vocab[sentBatch_text[j][j2]]
                --print("NB2")
            --else
                --encoderInputs[j2+eosOffset][j] = Vocab['<unknown>']
            --end

        end

        for j2=1,#sentBatch_smy[j]-1 do
            if Secab[sentBatch_smy[j][j2]] ~= nil then 
                decoderInputs[j2][j] = Secab[sentBatch_smy[j][j2]]
            else
                decoderInputs[j2][j] = Secab['<unknown>']
            end
        end

        for j2=2,#sentBatch_smy[j] do
            if Secab[sentBatch_smy[j][j2]] ~= nil then
                decoderTargets[j2-1][j] = Secab[sentBatch_smy[j][j2]]
            else
                decoderTargets[j2-1][j] = Secab['<unknown>']
            end
        end
    end
    

    if encoderInputs:dim()==0 then
        print("returning nextBatch -1")
        return -1
    end
    if encoderInputs:size()[1] > 100 then
        encoderInputs=encoderInputs:sub(-100,-1)
    end
    print("returning nextBatch")
    encoderInputs = encoderInputs:cuda()
    return encoderInputs, decoderInputs, decoderTargets
end 
--*************************************************************

function getVocab(filename)
    numDocumentsTraversed = 0
    local VocabSize = 0
    local Vocab = {}
    local Index2Vocab = {}
    VocabCnt = {}
    file_txt=io.open(filename,'r')

    --insert <eos>, <go> and <unknown> tokens

    Vocab['<go>'] = 1
    Vocab['<eos>'] = 2
    Vocab['<unknown>'] = 3
    Index2Vocab[1] = '<go>'
    Index2Vocab[2] = '<eos>'
    Index2Vocab[3] = '<unknown>'
    VocabSize = 3

    line = file_txt:read():lower()
    while line~=nil do
        if Vocab[line] == nil and line:match('.*[0-9]*.*') ~= nil then 
            VocabSize = VocabSize + 1
            Vocab[line] = VocabSize
            Index2Vocab[VocabSize] = line
            ----------change VocabCnt
            VocabCnt[line] = 1
        end
        
        --------change VocabCnt
        if Vocab[line] ~= nil and line:match('.*[0-9]*.*') ~= nil then 
            VocabCnt[line] = VocabCnt[line] + 1
        end
        
       line = file_txt:read()
       if (line ~= nil) then
           line=line:lower()
       end

       if line == "" then
           numDocumentsTraversed = numDocumentsTraversed +1
       end
    
       if numDocumentsTraversed > numDocuments then 
           break
       end
    end

    ---------------change VocabCnt
    VocabSize2 = 0
    Index2Vocab2 = {}
    Vocab2 = {}
    
    Vocab2['<go>'] = 1
    Vocab2['<eos>'] = 2
    Vocab2['<unknown>'] = 3
    Index2Vocab2[1] = '<go>'
    Index2Vocab2[2] = '<eos>'
    Index2Vocab2[3] = '<unknown>'
    VocabSize2 = 3
    
    for word, index in pairs(Vocab) do 
        if word ~= '<go>' and word ~= '<eos>' and word ~= '<unknown>' then
            if VocabCnt[word] >= minVocabCount then 
                VocabSize2 = VocabSize2 + 1
                Vocab2[word] = VocabSize2
                Index2Vocab2[VocabSize2] = word
            end
        end
    end
    Vocab = nil; Index2Vocab = nil 
    Vocab = Vocab2; Index2Vocab = Index2Vocab2 
    VocabSize = VocabSize2
    return Vocab, Index2Vocab, VocabSize
end

--*************************************************************



--*************************************************************
feval = function(x_new)
    --print("INSIDE feval")
    if x ~= x_new then
        x:copy(x_new)
    end
    
    dl_dx:zero()

    Inputs, _tmp, Targets = nextBatch()
    prediction = model:forward(Inputs)    
    Targets = Targets:transpose(1,2):select(2,1)

    local loss_x = criterion:forward(prediction, Targets)
    
    model:backward(Inputs, criterion:backward(prediction, Targets))

    if(_count%10 == 0) then
        _sen_txt = ""
        for _k = 1, #sent_txt_Inputs do 
            _sen_txt = _sen_txt .. sent_txt_Inputs[_k]
        end
        --print('______________________________________________')
        --print("SENTENCE:" .. _sen_txt )

        _, _tmp = prediction:select(1,batchSize):topk(1,1,true,true);
        _tmp =_tmp[1]
        
        _sen_target = ""
        for _i = 1, batchSize do
            _sen_target = _sen_target .." ".. Index2Secab[Targets[_i]]
        end

        if Index2Secab[_tmp] ~= nil then 
            print("".._sen_target..":::::"..Index2Secab[_tmp]);
        else
            print("TARGET:"..Targets[batchSize]..":::::PREDICTED:".._tmp);
        end

    end

    --print("returning FEVAL")
    return loss_x, dl_dx
end

--*************************************************************
--Get Vocab Table 
print("Getting Vocabulary...")
Vocab, Index2Vocab, VocabSize = getVocab('text_words.csv')
Secab, Index2Secab, SecabSize = getVocab('summary_words.csv')
wordVecSize = 300

sgd_params = {
    learningRate = 0.5, --changed from 1e-2
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0.2
}

print(wordVecSize, hiddenSize, SecabSize)
-- build simple recurrent neural network
model = nn.Sequential()
   :add(nn.Sequencer(nn.FastLSTM(wordVecSize,hiddenSize)))
   :add(nn.Sequencer(nn.Linear(hiddenSize,SecabSize)))
   :add(nn.Sequencer(nn.LogSoftMax()))
   :add(nn.Select(1,-1))  
--criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

model = model:cuda()
criterion = criterion:cuda()

x,dl_dx = model:getParameters()

for outerj = 1,40 do
    revcnt = 1
    sumcnt = 1
	file_txt = io.open('text_words.csv','r')
	file_smy = io.open('summary_words.csv','r')
    --print("--------------EPOCH"..outerj.."-------------------")
    for inneri = 1, numDocuments/batchSize do
        _count = inneri
        _, fs = optim.sgd(feval,x, sgd_params )
        model:forget()
        if inneri%10 == 0 then
            print('error for Batch ' .. sgd_params.evalCounter  .. ' is ' .. " Number Reviews : ".. sgd_params.evalCounter*batchSize .. " : ".. fs[1])
        end
    end
    if outerj%10 == 0  then 
        save_model = {}
        save_model['model'] = model
        save_model['Vocab'] = Vocab
        save_model['Index2Vocab'] = Index2Vocab
        save_model['Secab'] = Secab
        save_model['Index2Secab'] = Index2Secab
        str_save = 'Models/model' .. outerj
        torch.save(str_save, save_model)
    end
end

