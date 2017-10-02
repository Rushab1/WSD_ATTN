require 'rnn'
require 'optim'
require 'seq2seq'
require 'cutorch'
require 'cunn'
require 'attention'

---------------------------------
cmd = torch.CmdLine()
cmd:option('-model', 'Models/model13')
cmd:option('-testfile', 'test' )
cmd:option('-task', 'eval')
cmd:option('-confusionMatrix', 'False')
cmd:option('-confusionLabel', 'None')
opt = cmd:parse(arg)

if opt.testfile == 'test' then 
    f = io.open('test_summary_words.csv','r')
elseif opt.testfile == 'train' then
    f = io.open('./cv_summary_words.csv','r')
elseif opt.testfile == 'cvValid' then
    f = io.open('./cv_summary_words_valid.csv','r')
elseif opt.testfile == 'cvTest' then
    f = io.open('./cv_summary_words_test.csv','r')
else
    print("INVALID OPTION --testfile ['test', 'cvValid', 'cvTest'] " .. opt.testfile)
    os.exit()
end

numDocuments = 0
tmp = f:read()
while tmp ~= nil do
    numDocuments = numDocuments + 1
    tmp = f:read()
    if tmp == "" then 
        tmp = f:read()
    end
end

cutorch.setDevice(1)
torch.setdefaulttensortype('torch.CudaTensor')
----------------------------------

if opt.testfile == 'test' then 
    file_txt=io.open('test_text_words.csv','r')
    file_smy=io.open('test_summary_words.csv','r')
elseif opt.testfile == 'train' then
    file_txt=io.open('cv_text_words.csv','r')
    file_smy=io.open('./cv_summary_words.csv','r')
elseif opt.testfile == 'cvValid' then
    file_txt=io.open('cv_text_words_valid.csv','r')
    file_smy=io.open('./cv_summary_words_valid.csv','r')
elseif opt.testfile == 'cvTest' then
    file_txt=io.open('cv_text_words_test.csv','r')
    file_smy=io.open('./cv_summary_words_test.csv','r')
else
    print("INVALID OPTION --testfile ['test', 'cv'] " .. opt.testfile)
    os.exit()
end

--file_txt=io.open('tt','r')
--file_smy=io.open('ts','r')
batchSize = 1

--numDocuments = 100

--loaded_model = torch.load('Models/model13')
loaded_model = torch.load(opt.model)
model = loaded_model.model
attn = Attention.new()
attn.attention_weights = model.attention.attention_weights
Vocab = loaded_model.Vocab
Secab = loaded_model.Secab
Index2Vocab = loaded_model.Index2Vocab
Index2Secab = loaded_model.Index2Secab
SecabSize = #Index2Secab

-----------------------SET DROPOUT TO ZERO-----------------------------------
model.encoder.modules[1].modules[1].modules[3].p = 0
-----------------------------------------------------------------------------

--print(model)
--loaded_model = torch.load('lowermodel_tmp')
--lowermodel = loaded_model.model
--lowerVocab = loaded_model.Vocab
--lowerSecab = loaded_model.Secab
--lowerIndex2Vocab = loaded_model.Index2Vocab 
--lowerIndex2Secab = loaded_model.Index2Secab
--lowerSecabSize = #lowerIndex2Secab
--*************************************************************

function nextBatch()
    local encoderInputs, decoderInputs, decoderTargets = {}, {}, {}

    sentBatch_text = {}
    sentBatch_smy={}
    local max_len_text = 0
    local max_len_smy = 0 
    local j=0, j2

    for j=1,batchSize do 
        local sent_txt = {}
        local sent_smy = {}

        line = file_txt:read():lower()
        while line ~= "" and line~=nil do 
            table.insert(sent_txt,line)
            line = file_txt:read()
            if line ~= nil then
                line = line:lower()
            end
        end
        table.insert(sentBatch_text,sent_txt)
        
        if max_len_text < #sent_txt then
            max_len_text = #sent_txt
        end

        table.insert(sent_smy, '<go>')
        line = file_smy:read():lower()
        while line ~= "" and line~= nil  do    
            table.insert(sent_smy, line)
            line = file_smy:read()
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
    

    local encoderInputs = torch.IntTensor(max_len_text, batchSize):fill(0)
    local decoderInputs = torch.IntTensor(max_len_smy-1,batchSize):fill(0)
    local decoderTargets= torch.IntTensor(max_len_smy-1,batchSize):fill(0)

    for j=1,batchSize do
        eosOffset = max_len_text - #sentBatch_text[j] --left paddding

        for j2=1,#sentBatch_text[j]do   
            if  Vocab[sentBatch_text[j][j2]] ~= nil then 
                encoderInputs[j2+eosOffset][j] = Vocab[sentBatch_text[j][j2]]
            else
                --encoderInputs[j2+eosOffset][j] = Vocab['<unknown>']
                encoderInputs[j2+eosOffset][j] = 0
            end

        end

        --trimmedEosToken = sentBatch_smy[j]:sub(1,-2)
        for j2=1,#sentBatch_smy[j]-1 do
            if Secab[sentBatch_smy[j][j2]] ~= nil then 
                decoderInputs[j2][j] = Secab[sentBatch_smy[j][j2]]
            else
                decoderInputs[j2][j] = Secab['<unknown>']
            end
        end
        

        --trimmedGoToken = sentBatch_smy[j]:sub(2,-1)
        for j2=2,#sentBatch_smy[j] do
            if Secab[sentBatch_smy[j][j2]] ~= nil then
                decoderTargets[j2-1][j] = Secab[sentBatch_smy[j][j2]]
            else
                decoderTargets[j2-1][j] = Secab['<unknown>']
            end
        end
    end
    

    if encoderInputs:dim()==0 then
        return -1
    end

    if encoderInputs:size()[1] > 100 then
        encoderInputs=encoderInputs:sub(-100,-1)
    end

    len1 = encoderInputs:size()[1]
    len2 = encoderInputs:size()[2]
    encoderInputsReverse = torch.IntTensor(encoderInputs:size()):fill(0)

    f = torch.IntTensor(2,len1,len2)
    for _i = 1, len1 do    
        for _j = 1, len2 do 
            --encoderInputsReverse[_i][_j] = encoderInputs[len1-_i+1][_j]
            f[1][_i][_j] = encoderInputs[_i][_j]
            f[2][_i][_j] = encoderInputs[len1-_i+1][_j]
        end
    end

    e = {}
    e[1] = encoderInputs
    e[2] = encoderInputsReverse 

    f = f:cuda()
    return f, {decoderInputs}, {decoderTargets}
    --return {encoderInputs}, {decoderInputs}, {decoderTargets}, sentBatch_smy
end 
--*************************************************************


--*************************************************************
eval = function()
    acnt = 0
    ccnt = 0
    evalText = io.open('evalText.txt','w')
    evalUserSmy = io.open('target_summary.txt','w')
    evalPredSmy = io.open('predicted_summary.txt','w')
    noPredCnt = 0

    print("Evaluating " .. numDocuments)
    for i = 1,numDocuments/batchSize do
        model.encoder:forget()
        model.decoder:forget()
        Inputs, decoderInputs, Targets, sentBatch_smy = nextBatch()
       
        for batchSize_ind = 1,batchSize do
            str_sent = "" 
            enc_batch_size = Inputs[1]:select(2,batchSize_ind)

            --Write review to file
            for ii = 1,(#enc_batch_size)[1] do   
                if enc_batch_size[ii] ~= 0 then
                    local word = Index2Vocab[enc_batch_size[ii]]
                    str_sent = str_sent .. " " .. word
                end
            end 
            evalText:write(str_sent .. '\n')
            review_text = str_sent

            --Write user summary to file 
            str_sent = "" 
            decoder_batch_size = Targets[1]:select(2,batchSize_ind)

            for ii = 1,(#decoder_batch_size)[1] do
                if decoder_batch_size[ii] ~= 0 then
                    local word = Index2Secab[decoder_batch_size[ii]]
                    str_sent = str_sent .. " " .. word
                end
            end 
            
            evalUserSmy:write(Index2Secab[decoder_batch_size[1]] .. "\n")
            user_text = Index2Secab[decoder_batch_size[1]]

            --Predicted summary 
            encoderOutput = model.encoder:forward(Inputs)
            attn:__init__(encoderOutput)
            attn.attention_weights = model.attention.attention_weights
            attnOutput = attn:forward(encoderOutput)
            probabilities = model.decoder:forward(attnOutput)

            probabilities:resize(SecabSize)
            probabilities, wordIds = probabilities:topk((#probabilities)[1], 1, true, true)
            str_sent = Index2Secab[wordIds[1]]
            
            if str_sent == "" then 
                noPredCnt = noPredCnt + 1
            end

            if lowermodel ~= nil then 
                probabilities2 = lowermodel:forward(Inputs[1])
                probabilities2:resize(lowerSecabSize)
                probabilities2, wordIds2 = probabilities2:topk((#probabilities2)[1], 1, true, true)
            end

            pred_text = str_sent

            check = 0

            if not string.match(user_text, pred_text) then 
                ccnt = ccnt + 1
                if check == 0 then 
                    time_waste  = 1
                else
                    time_waste  = 1
                end
            end
            evalPredSmy:write(pred_text .. "\n")
        end
    end
    evalText:close()
    evalUserSmy:close()
    evalPredSmy:close()
    cmd = "python quantify_results.py"
    if opt.confusionMatrix == "True" then
        cmd = cmd .. " --confusionMatrix " .. opt.confusionLabel
    end
    os.execute(cmd) 
    print("\nacnt = " .. acnt .. " : ccnt = " .. ccnt .. "/" .. numDocuments )
end

--*************************************************************

predict = function(text)
    local text = text:split(' ')
    if #text > 100 then
        print('Too big a sentence')
        return
    end

    local Inputs, InputsRev = torch.Tensor(#text), torch.Tensor(#text)


    for ii=1,#text do
        text[ii] = text[ii]:lower()
        if Vocab[text[ii]] ~= nil then
            Inputs[ii] = Vocab[text[ii]]
        else
            Inputs[ii] = 0
            print('Unknown words present in the sentence. Aborting : '..text[ii])
        end
    end
    
    ip = torch.Tensor(2, #text,1)

    for i = 1, #text do
        ip[1][i] = Inputs[i]
        ip[2][i] = Inputs[#text - i + 1]
    end

    ip = ip:cuda()

    encoderOutput = model.encoder:forward(ip)
    attnOutput = model.attn:forward(encoderOutput)
    probabilities = model.decoder:forward(attnOutput)


    probabilities:resize(SecabSize)
    probabilities, wordIds = probabilities:topk((#probabilities)[1], 1, true, true)
    str_sent = Index2Secab[wordIds[1]]

    print(str_sent)
end


--*************************************************************
if opt.task == 'predict' then
    predict("START START It was very hard to do things in the ")
    predict("START START I have no hard feelings for anyone . END")
    predict("START The surface was so hard that we could not run")
else
    eval()
end
