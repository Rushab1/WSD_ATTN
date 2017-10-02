require 'rnn'
require 'optim'
require 'seq2seq'
require 'cutorch'
require 'cunn'
---------------------------------

cutorch.setDevice(1)
torch.setdefaulttensortype('torch.CudaTensor')
print('Running with CUDA on GPU')
----------------------------------

file_txt=io.open('text_words.csv','r')
file_smy=io.open('summary_words.csv','r')

batchSize = 70
numDocuments = 6000
hiddenSize = 300
--minVocabCount = 4
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
    sentBatch_text = {}
    sentBatch_smy={}
    local max_len_text = 0
    local max_len_smy = 0 
    local j=0, j2

    -------------------------------------------CHANGE------------------------------
    local checkword = {}
    local checksize = {}
    for _ind,_word in ipairs(SenseWords) do 
        checksize[_word] = 0
    end 
    -------------------------------------------TILL HERE------------------------------

    for j=1,batchSize do 

        local sent_txt = {}
        local sent_smy = {}

        line = reviews[revcnt]; revcnt = revcnt + 1

        while line ~= "" and line~=nil do 
            table.insert(sent_txt,line)
            line = reviews[revcnt]; revcnt = revcnt + 1

            -------------------------------------------CHANGE------------------------------
            if InverseSenseWords[line] ~= nil then 
                table.insert(checkword,line)
                checksize[line] = checksize[line] + 1
            end
            -------------------------------------------TILL HERE---------------------------
                
            if line ~= nil then
                line = line:lower()
            end
        end
        table.insert(sentBatch_text,sent_txt)
        
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
    
        table.insert(sent_smy, '<eos>')
        table.insert(sentBatch_smy,sent_smy)
        
        if max_len_smy < #sent_smy then
            max_len_smy = #sent_smy
        end 
    end 
    
    local encoderInputs = {}
    local decoderInputs = {}
    local decoderTargets = {}
    for _ind,_word in ipairs(SenseWords) do
        encoderInputs[_word] = torch.IntTensor(max_len_text, checksize[_word]):fill(0)
        decoderInputs[_word] = torch.IntTensor(max_len_smy-1,checksize):fill(0)
        decoderTargets[_word] = torch.IntTensor(max_len_smy-1,checksize):fill(0)
    end


    for j=1,batchSize do
        eosOffset = max_len_text - #sentBatch_text[j] --left paddding
        _word = checkword[j]

        for j2=1,#sentBatch_text[j]do 
            if  Vocab[sentBatch_text[j][j2]] ~= nil then 
                encoderInputs[_word][j2+eosOffset][j] = Vocab[sentBatch_text[j][j2]]
            else
                encoderInputs[_word][j2+eosOffset][j] = Vocab['<unknown>']
            end
        end

        for j2=1,#sentBatch_smy[j]-1 do
            if Secab[sentBatch_smy[j][j2]] ~= nil then 
                decoderInputs[_word][j2][j] = Secab[sentBatch_smy[j][j2]]
            else
                decoderInputs[_word][j2][j] = Secab['<unknown>']
            end
        end

        for j2=2,#sentBatch_smy[j] do
            if Secab[sentBatch_smy[j][j2]] ~= nil then
                decoderTargets[_word][j2-1][j] = Secab[sentBatch_smy[j][j2]]
            else
                decoderTargets[_word][j2-1][j] = Secab['<unknown>']
            end
        end
    end 
    
    local _cnt = 0
    for _ind, _word in ipairs(SenseWords) do 
        if encoderInputs[_word]:dim()==0 then
            _cnt = _cnt + encoderInputs[_word]:dim()
        end

        if encoderInputs[_word]:size()[1] > 150 then
            encoderInputs[_word]=encoderInputs[_word]:sub(-150,-1)
        end
    end

    if check == 0 then 
        return -1
    end 

    return encoderInputs, decoderInputs, decoderTargets
end 
--*************************************************************


--*************************************************************
feval = function(x_new,model)
    if x ~= x_new then
        x:copy(x_new)
    end
     
    for _ind,_word in ipairs(SenseWords) do 
        dl_dx[_word]:zero()
    end 

    --print('----getting NextBatch()')
    encoderInputs, decoderInputs, decoderTargets = nextBatch()
    --print('----Done getting NextBatch()')

    for _ind, _word in ipairs(SenseWords) do 
        --forward pass
        --print('----encoder forward')
        encoderOutput = model[_word].encoder:forward(encoderInputs[_word])
        --print('----DONE encoder forward')

        --print('----forwardConnect')
        model[_word]:forwardConnect(encoderInputs[_word]:size(1))
        --print('----DONE encoder forward')

        --print('----decoder forward')
        decoderOutput = model[_word].decoder:forward(decoderInputs[_word])
        --print('----DONE decoder forward')

        --print('----criterion forward')
        local loss_x = model[_word].criterion:forward(decoderOutput, decoderTargets[_word])
        --print('----DONE criterion forward')
       
        --backward pass
        --print('----criterion backward')
        dloss_doutput = model[_word].criterion:backward(decoderOutput, decoderTargets[_word])
        --print('----DONE criterion backward')

        --print('----decoder backward')
        model[_word].decoder:backward(decoderInputs[_word], dloss_doutput)
        --print('----DONE decoder backward')

        --print('----backwardConnect')
        model[_word]:backwardConnect()
        --print('----DONE backwardConnect')
     
        
        --print('----encoder backward')
        model[_word].encoder:backward(encoderInputs[_word], encoderOutput:zero())
        --print('----DONE encoder backward') 

        if sgd_params.evalCounter%10 == 0 then
            _tmp_rnd = (torch.floor(torch.rand(1)*#SenseWords)+1)[1]
            _tmp_word = SenseWords[_tmp_rnd]
            --print example output Review from Training data 
            print('\n___________________________________________')
            str_sent = "" 
            enc_batch_size = encoderInputs[_tmp_word]:select(2,1)
            for ii = 1,(#enc_batch_size)[1] do
                if enc_batch_size[ii] ~= 0 then
                    local word = Index2Vocab[enc_batch_size[ii]]
                    str_sent = str_sent .. " " .. word
                end
            end

            print("REVIEW:\n" .. str_sent .. '\n........................')
            
            wordIds, probabilities = model[_tmp_word]:eval(encoderInputs[_tmp_word]:select(2,1),_tmp_word,Secab[_tmp_word],Index2Secab[_tmp_word],SecabSize[_tmp_word])
            str_sent=""
            
            str_sent_target = ""
            
            tmp = 1
            tmp_err = 0 
            for _tmp, ind in ipairs(wordIds) do 
                local word = Index2Secab[_tmp_word][ind[1]]
                str_sent = str_sent .." ".. word

                if tmp > (#decoderTargets[_tmp_word])[1] then 
                    expected_output = '<out-of-range>'
                elseif decoderTargets[_tmp_word][tmp][1] ==0 then
                    expected_output = '<0-index>'
                else
                    --@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    expected_output = Index2Secab[_tmp_word][decoderTargets[_tmp_word][tmp][1]]
                    str_sent_target = str_sent_target .. " " .. expected_output
                end

                
                if expected_output == Index2Secab[_tmp_word][ind[1]] then
                    tmp_err = tmp_err + 1
                end


                tmp = tmp + 1
            end

            str_sent_target=""
            for i =1,(#decoderTargets[_tmp_word])[1] do
                if decoderTargets[_tmp_word][i][1] ~= 0 then
                    expected_output = Index2Secab[_tmp_word][decoderTargets[_tmp_word][i][1]]
                    str_sent_target = str_sent_target .. " " .. expected_output
                end
            end

            print("ACTUAL SUMMARY    : " .. str_sent_target .. "\nPREDICTED SUMMARY : " .. str_sent .. " : " .. loss_x .. " : " .. tmp_err/(tmp-1)*100)
        end
    end

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

--Get Vocab Table 
print("Getting Vocabulary...")

Vocab, Index2Vocab, VocabSize = getVocab('text_words.csv')
--Secab, Index2Secab, SecabSize = getVocab('summary_words.csv')

--Manually creating Secab 
Secab = {}
Index2Secab = {}
SecabSize = {}
SenseWords = {'line','hard','interest'}
InverseSenseWords  = {}

for _ind, _word in ipairs(SenseWords) do 
    InverseSenseWords[_word] = 1
    Secab[_word]={}
end
------------------------------------model built--------------------------------
Index2Secab['line'] = {'cord','phone','division','text','formation','product'}
Index2Secab['hard'] = {'hard1','hard2','hard3'}
Index2Secab['interest'] = {'interest1','interest2','interest3','interest4','interest5','interest6'}

SecabSize['line'] = 6
SecabSize['hard'] = 3
SecabSize['interest'] = 6

Secab['line']['cord'] = 1
Secab['line']['phone'] = 2
Secab['line']['division'] = 3
Secab['line']['text'] = 4
Secab['line']['formation'] = 5
Secab['line']['product'] = 6

Secab['hard']['hard1'] = 1
Secab['hard']['hard2'] = 2
Secab['hard']['hard3'] = 3

Secab['interest1'] = 1
Secab['interest2'] = 2
Secab['interest3'] = 3
Secab['interest4'] = 4
Secab['interest5'] = 5
Secab['interest6'] = 6
--Done creating Secab

print('Text Vocabulary Size : ' .. VocabSize)
print('Sense Vocabulary Size : ')
print(SecabSize)
print("Done")

print('Building Model')
model = {}

for _ind, _word in ipairs(SenseWords) do 
    model[_word] = Seq2Seq(VocabSize,SecabSize[_word],hiddenSize)
    model[_word].goToken = Vocab['<go>']
    model[_word].eosToken = Vocab['<eos>']

    if batchSize > 1 then 
        model[_word].criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))
    else
        model[_word].criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
    end

    model[_word]:cuda()
end 

print('Done')
------------------------------------model built--------------------------------

-- get weights and gradient of loss wrt weights from the model
print('Getting Parameters')
x, dl_dx = {}, {}
for _ind, _word in ipairs(SenseWords) do 
    x[_word], dl_dx[_word] = model[_word]:getParameters()
end 
print('Done')

print('Getting sgd_params')
sgd_params = {
    learningRate = 0.5,--changed from 1e-2
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0.5
}
print('Done')

print('Training')
str_error = "batchSize : " .. batchSize .. "; Reviews per Epoch " .. numDocuments .. ";\nMean -> Median -> Variance\n"
for j=1,10000 do
    print("------------EPOCH: "..j.."---------------")

    revcnt = 1
    sumcnt = 1
	file_txt = io.open('text_words.csv','r')
	file_smy = io.open('summary_words.csv','r')

    epoch_misclassificationError = torch.FloatTensor(numDocuments/batchSize)
    epoch_NLLError = torch.FloatTensor(numDocuments/batchSize)

    for i = 1, numDocuments/batchSize do	
        _, fs = optim.sgd(feval,x, sgd_params)
        
        for _ind,_word in ipairs(SenseWords) do 
            model[_word].decoder:forget()
            model[_word].encoder:forget()
        end

        print('error for Batch ' .. sgd_params.evalCounter  .. ' is ' .. " Number Reviews : ".. sgd_params.evalCounter*batchSize .. " : ".. fs[1])
        str_error = str_error .. fs[1] ..'\n'
        epoch_NLLError[i] = fs[1]
    end

        print("NLL Error Mean     : " .. torch.mean(epoch_NLLError))
        print("NLL Error Median   : " .. torch.median(epoch_NLLError)[1])
        print("NLL Error Std Dev  : " .. torch.std(epoch_NLLError))
       

        error_file = io.open('Models/error_log',"w")
        error_file:write(str_error)
        error_file:close()

        if j%3==0 then 
            save_model = {}
            save_model['model'] = model
            save_model['Vocab'] = Vocab
            save_model['Index2Vocab'] = Index2Vocab
            save_model['Secab'] = Secab
            save_model['Index2Secab'] = Index2Secab
            str_save = 'Models/model' .. sgd_params.evalCounter*batchSize
            torch.save(str_save, save_model)
        end
end
print('Done')
