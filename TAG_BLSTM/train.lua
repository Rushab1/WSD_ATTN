require 'rnn'
require 'optim'
require 'seq2seq'
require 'cutorch'
require 'cunn'
require './generalFunctions.lua'

<<<<<<< HEAD
cmd = torch.CmdLine()
cmd:option('-eval', 'False')
opt = cmd:parse(arg)

f = io.open('cv_summary_words.csv','r')
=======
f = io.open('summary_words.csv','r')
>>>>>>> parent of d1025b3... Temporary
numDocuments = 0
tmp = f:read()
while tmp ~= nil do
    numDocuments = numDocuments + 1
    tmp = f:read()
    if tmp == "" then 
        tmp = f:read()
    end
end

print("Number of Training examples = " .. numDocuments)
--file_txt=io.open('text_words.csv','r')
--file_smy=io.open('summary_words.csv','r')

<<<<<<< HEAD
batchSize = 20
=======
batchSize = 7
>>>>>>> parent of d1025b3... Temporary
--numDocuments = 9350
hiddenSize = 150
hiddenSizeTag = 5
--hiddenSize = 50
minVocabCount = 0
print('Begin')

--*************************************************************

reviews = {}
reviewstags = {}
summaries = {}

rf = io.open('text_words.csv','r')
tf = io.open('text_words_tags.csv', 'r')
sf = io.open('summary_words.csv','r')

cnt1 = 0
cnt2 = 0
rev = rf:read():lower()
while rev ~= nil do
    table.insert(reviews, rev:lower())
    rev = rf:read()
    cnt1 = cnt1  + 1
end

revtag = tf:read():lower()
while revtag ~= nil do
    table.insert(reviewstags, revtag:lower())
    revtag = tf:read()
    cnt2 = cnt2 + 1
end

sum = sf:read():lower()
while sum ~= nil do
    table.insert(summaries, sum:lower())
    sum = sf:read()
end

print(cnt1,cnt2)
revcnt = 1
sumcnt = 1 

function nextBatch()

    sentBatch_text = {}
    sentBatch_tags = {}
    sentBatch_smy = {}
    local max_len_text = 0
    local max_len_tags = 0
    local max_len_smy = 0 
    local j = 0, j2

    for j = 1, batchSize do 
        local sent_txt = {}
        local sent_tags = {}
        local sent_smy = {}

        --line = file_txt:read():lower()
        linetags = reviewstags[revcnt];
        line = reviews[revcnt]; revcnt = revcnt + 1

        while line ~= "" and line~=nil do 
            table.insert(sent_txt,line)
            table.insert(sent_tags,linetags)
            --line = file_txt:read()
            linetags = reviewstags[revcnt]
            line = reviews[revcnt]; revcnt = revcnt + 1
            if line ~= nil then
                line = line:lower()
                linetags = linetags:lower()
            end
        end
        table.insert(sentBatch_text,sent_txt)
        table.insert(sentBatch_tags,sent_tags)
        
        if max_len_text < #sent_txt then
            max_len_text = #sent_txt
        end

        table.insert(sent_smy, '<go>')
        line = summaries[sumcnt]; sumcnt = sumcnt + 1
        local check = 0
        while line ~= "" and line~= nil  do     
            check = 1
            table.insert(sent_smy, line)
            line = summaries[sumcnt]; sumcnt = sumcnt + 1
            if line~=nil then
                line = line:lower()
            end
        end

        table.insert(sent_smy, '<eos>')
        table.insert(sentBatch_smy,sent_smy)
        
        if max_len_smy < #sent_smy then
            max_len_smy = #sent_smy
        end 
    end 

    local encoderInputs  = torch.IntTensor(max_len_text,  batchSize):fill(0)
    local encoderTags  = torch.IntTensor(max_len_text,  batchSize):fill(0)
    local decoderInputs  = torch.IntTensor(max_len_smy-1, batchSize):fill(0)
    local decoderTargets = torch.IntTensor(max_len_smy-1, batchSize):fill(0)

    for j = 1, batchSize do
        eosOffset = max_len_text - #sentBatch_text[j] --left paddding

        for j2=1,#sentBatch_text[j]do   
            if  Vocab[sentBatch_text[j][j2]] ~= nil then 
                encoderInputs[ j2 + eosOffset ][j] = Vocab   [sentBatch_text[j][j2]]
                encoderTags[   j2 + eosOffset ][j] = Tocab[sentBatch_tags[j][j2]]
            else
                encoderInputs[ j2 + eosOffset ][j] = Vocab['<unknown>']
                encoderTags  [ j2 + eosOffset ][j] = Tocab['<unknown>']
            end
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
                unknown_target = sentBatch_smy[j][j2] 
                decoderTargets[j2-1][j] = Secab['<unknown>']
            end
        end
    end
    
    if encoderInputs:dim()==0 then
        return -1
    end


    if encoderInputs:size()[1] > 100 then
        encoderInputs = encoderInputs:sub(-100, -1)
        encoderTags = encoderTags:sub(-100, -2)
    end

    --encoderInputs = encoderInputs:resize(encoderInputs:size()[1])
    --decoderInputs = decoderInputs:resize(decoderInputs:size()[1])
    --decoderTargets = decoderTargets:resize(decoderTargets:size()[1])
    
    len1 = encoderInputs:size()[1]
    len2 = encoderInputs:size()[2]
    encoderInputsReverse = torch.IntTensor(encoderInputs:size()):fill(0)
    encoderTagsReverse = torch.IntTensor(encoderTags:size()):fill(0)

    
    f = torch.IntTensor(2,len1,len2)
    g = torch.IntTensor(2,len1,len2)
    for _i = 1, len1 do    
        for _j = 1, len2 do 
            --encoderInputsReverse[_i][_j] = encoderInputs[len1-_i+1][_j]
            f[1][_i][_j] = encoderInputs[_i][_j]
            f[2][_i][_j] = encoderInputs[len1-_i+1][_j]
            g[1][_i][_j] = encoderTags[_i][_j]
            g[2][_i][_j] = encoderTags[len1-_i+1][_j]
        end
    end

    e = {}
    e[1] = encoderInputs
    e[2] = encoderInputsReverse
    eTags = {}
    eTags[1] = encoderTags
    eTags[2] = encoderTagsReverse

    f = f:cuda()
    g = g:cuda()

    h = torch.IntTensor(2, 2, len1, len2)
    for _i = 1, len1 do    
        for _j = 1, len2 do 
            --encoderInputsReverse[_i][_j] = encoderInputs[len1-_i+1][_j]
            h[1][1][_i][_j] = encoderInputs[_i][_j]
            h[1][2][_i][_j] = encoderInputs[len1-_i+1][_j]
            h[2][1][_i][_j] = encoderTags[_i][_j]
            h[2][2][_i][_j] = encoderTags[len1-_i+1][_j]
        end
    end

    h = h:cuda()
    return h, f, g, {decoderInputs}, {decoderTargets}
    --return encoderInputs, decoderInputs, decoderTargets
end 
--*************************************************************

--*************************************************************
feval = function(x_new)
    if x ~= x_new then
        x:copy(x_new)
    end
    dl_dx:zero()
    --Inputs, Tags, decoderInputs, Targets = nextBatch()
    Inputs, Sent, Tags, decoderInputs, Targets = nextBatch()
    
    --Output = model:forward(Inputs[1])
    Output = model:forward(Inputs)
    dt = torch.IntTensor(batchSize,1)

    for _i = 1, batchSize do
        _t= Targets[1][1][_i]
        dt[_i][1] = _t
    end

    loss_x = criterion:forward(Output, dt)

    dloss_doutput = criterion:backward(Output, dt)

    --b = model:backward(Inputs[1],dloss_doutput)
    b = model:backward(Inputs,dloss_doutput)

    if sgd_params.evalCounter%10 == 0 then
        --str_sent = "" 
        --enc_batch_size = Inputs[1]:select(2,1)
        --for ii = 1,(#enc_batch_size)[1] do
            --if enc_batch_size[ii] ~= 0 then
                --local word = Index2Vocab[enc_batch_size[ii]]
                --str_sent = str_sent .. " " .. word
            --end
        --end

        probabilities = Output:select(1,1)
        probabilities:resize(SecabSize)
        probabilities, wordIds = probabilities:topk((#probabilities)[1], 1, true, true)

        str_sent = Index2Secab[wordIds[1]]

        str_sent_target = Index2Secab[Targets[1][1][1]]
        
        print("ACTUAL SUMMARY    : " .. str_sent_target .. "PREDICTED SUMMARY : " .. str_sent )
        model:forget()
    end

    model:forget()
    return loss_x, dl_dx
end

--*************************************************************

function getVocab(filename)
    numDocumentsTraversed = 0
    local VocabSize = 0
    local Vocab = {}
    local Index2Vocab = {}
    VocabCnt = {}
    file_txt=io.open(filename,'r')

    Vocab['<unknown>'] = 1
    Index2Vocab[1] = '<unknown>'
    VocabSize = 1

    line = file_txt:read():lower()
    while line~=nil do
        if Vocab[line] == nil and line:match('.*[0-9]*.*') ~= nil then 
            VocabSize = VocabSize + 1
            Vocab[line] = VocabSize
            Index2Vocab[VocabSize] = line
            VocabCnt[line] = 1
        end
        
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

    VocabSize2 = 0
    Index2Vocab2 = {}
    Vocab2 = {}
    
    Vocab2['<unknown>'] = 1
    Index2Vocab2[1] = '<unknown>'
    VocabSize2 = 1
    
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
print("Getting Vocabulary...")

Vocab, Index2Vocab, VocabSize = getVocab('text_words.csv')
Tocab, Index2Tocab, TocabSize = getVocab('text_words_tags.csv')
print("+++++++++++++++++++++++",Tocab)
Secab, Index2Secab, SecabSize = getVocab('summary_words.csv')

print('Text Vocabulary Size : ' .. VocabSize)
print('Sense Vocabulary Size : ' .. SecabSize)
print(Index2Secab)
print("Done")

print('Building Model')

model1 = nn.Sequential()
      --:add(nn.SelectTable(1))
      :add(nn.Select(1,1))
      :add(nn.Sequencer(nn.LookupTableMaskZero(VocabSize, hiddenSize)))
      :add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize)))
      --:add(nn.Sequencer(nn.Linear(hiddenSize,SecabSize)))

model2 = nn.Sequential()
      --:add(nn.SelectTable(2))
      :add(nn.Select(1,2))
      :add(nn.Sequencer(nn.LookupTableMaskZero(VocabSize, hiddenSize)))
      :add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize)))
      --:add(nn.Sequencer(nn.Linear(hiddenSize,SecabSize)))

modeljoin = nn.Concat(3)
modeljoin:add(model1)
modeljoin:add(model2)

newmodel1 = nn.Sequential()
          :add(nn.Select(1,1))
          :add(modeljoin)
---------------------------------------------------------------

tagmodel1 = nn.Sequential()
      --:add(nn.SelectTable(1))
      :add(nn.Select(1,1))
      :add(nn.Sequencer(nn.LookupTableMaskZero(TocabSize, hiddenSizeTag)))
      :add(nn.Sequencer(nn.FastLSTM(hiddenSizeTag, hiddenSizeTag)))
      --:add(nn.Sequencer(nn.Linear(hiddenSizeTag,SecabSize)))

tagmodel2 = nn.Sequential()
      --:add(nn.SelectTable(1))
      :add(nn.Select(1,2))
      :add(nn.Sequencer(nn.LookupTableMaskZero(TocabSize, hiddenSizeTag)))
      :add(nn.Sequencer(nn.FastLSTM(hiddenSizeTag, hiddenSizeTag)))
      --:add(nn.Sequencer(nn.Linear(hiddenSizeTag,SecabSize)))

modeljointag = nn.Concat(3)
             :add(tagmodel1)
             --:add(tagmodel2)

newmodel2 = nn.Sequential()
         :add(nn.Select(1,2))
         :add(modeljointag)

supermodeljoin = nn.Concat(3)
               :add(newmodel1)
               :add(newmodel2)

model = nn.Sequential()
      :add(supermodeljoin) 
      --:add(newmodel1)
      :add(nn.Sequencer(nn.Linear(2*hiddenSize + hiddenSizeTag,SecabSize)))
      --:add(nn.Sequencer(nn.Linear(2*hiddenSize,SecabSize)))
      :add(nn.Sequencer(nn.LogSoftMax()))
      :add(nn.Select(1,-1))

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
checktable = { torch.floor(torch.rand(20,1)*VocabSize) + 1 }
model:cuda()
criterion:cuda()
print('Done')

print('Getting Parameters')
x, dl_dx = model:getParameters()
print('Done')

print('Getting sgd_params')
sgd_params = {
    learningRate = 0.005, --changed from 1e-2
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0.2    -- changed from 0.2
}
print('Done')

print('Training')
str_error = "batchSize : " .. batchSize .. "; Reviews per Epoch " .. numDocuments .. ";\nMean -> Median -> Variance\n"

epochAccuracies = {}
avgEpochAccuracies = {}
stableCnt = 4 -- for how many epochs should the validation accuracy decrease

if opt.eval == 'True' then 
    f = io.open('testResults', 'a')
    f:write("\n\n")
    f:close()
    f = io.open('validResults', 'a')
    f:write("\n\n")
    f:close()
    f = io.open('OVRResultsLR', 'a')
    f:write("\n\n")
    f:close()
    f = io.open('testResultsSVC', 'a')
    f:write("\n\n")
    f:close()
end

for j=1, 100 do
    print("------------EPOCH: "..j.."---------------")

    print(#reviews, #reviewstags, #summaries)
    tmp1, tmpTags, tmp2 = taggedCombinedRandomShuffle(reviews, reviewstags, summaries)
    reviews = nil
    reviewstags = nil
    summaries = nil
    reviews = localCopy(tmp1)
    reviewstags = localCopy(tmpTags)
    summaries = localCopy(tmp2)

    revcnt = 1
    sumcnt = 1
    --file_txt = io.open('text_words.csv','r')
    --file_smy = io.open('summary_words.csv','r')

    for i = 1, numDocuments/batchSize do	
        _, fs = optim.sgd(feval,x, sgd_params)        
        model:forget()

        if i%10 == 0 then
            print('error for Batch ' .. sgd_params.evalCounter  .. ' is ' .. " Number Reviews : ".. sgd_params.evalCounter*batchSize .. " : ".. fs[1])
        end
        
--        print(i % math.floor(numDocuments/(4*batchSize)), i, math.floor(numDocuments/(4*batchSize)), numDocuments, batchSize, numDocuments/(4*batchSize))

        if i % math.floor(numDocuments/(batchSize)) == 0 then 
            save_model = {}
            save_model['model'] = model
            save_model['Vocab'] = Vocab
            save_model['Index2Vocab'] = Index2Vocab
            save_model['Tocab'] = Tocab
            save_model['Index2Tocab'] = Index2Tocab
            save_model['Secab'] = Secab
            save_model['Index2Secab'] = Index2Secab
            str_save = 'Models/model' .. j .."_"..  i
            torch.save(str_save, save_model)
        end
    end

    if opt.eval == 'True' then
        os.execute('th eval.lua -model Models/model' .. j .. ' -testfile cvValid')
        os.execute('python quantify_results.py --file trainAcc')
        os.execute('python quantify_results.py --file validResults --label ' .. j)
        os.execute('th eval.lua -model Models/model' .. j .. ' -testfile cvTest')
        os.execute('python quantify_results.py --file testResults --label ' .. j)

        f = io.open('trainAcc', 'r')
        tmpAcc = tonumber(f:read())
        table.insert(epochAccuracies, tmpAcc)
        table.insert(avgEpochAccuracies, tmpAcc)
        if j > 2 then
            avgEpochAccuracies[j-1] = (epochAccuracies[j-2] + epochAccuracies[j-1] + epochAccuracies[j])/3
        end
        print("______________________________")
        print(avgEpochAccuracies)
        print('-----------------------')
        print(epochAccuracies)
        f:close()
        f = io.open('trainAcc', 'w')
        f:write("")
        f:close()

        --One vs rest on SentVectors
        --cmd = 'th genSentVec.lua -model Models/model' .. j .. ' -set cvtrain'
        --os.execute(cmd)
        --cmd = 'th genSentVec.lua -model Models/model' .. j .. ' -set cvtest'
        --os.execute(cmd)
        --cmd = 'python OVR.py ' .. j
        --os.execute(cmd)
    end
end

if opt.eval == 'True' then 
    f = io.open('avgEpochAccuracies' , 'a')
    f:write("\n\n")
    for i,j in ipairs(avgEpochAccuracies) do 
        f:write(i .. " " .. j .. "\n")
    end
end

print('Done')

