require 'rnn'
require 'optim'
require 'seq2seq'
require 'cutorch'
require 'cunn'

if w2vutils == nil then 
    w2vutils = require("w2vutils.lua")
end
---------------------------------

--cmd = torch.CmdLine()
--cmd:option('-model','model.t7')
--opt = cmd:parse(arg)

cutorch.setDevice(1)
torch.setdefaulttensortype('torch.CudaTensor')
----------------------------------

file_txt=io.open('test_text_words.csv','r')
file_smy=io.open('test_summary_words.csv','r')

batchSize = 50
numDocuments = 130

--loaded_model = torch.load('model_tmp')
loaded_model = torch.load('./model_tmp')
model = loaded_model.model
Vocab = loaded_model.Vocab
Secab = loaded_model.Secab
Index2Vocab = loaded_model.Index2Vocab 
Index2Secab = loaded_model.Index2Secab

--*************************************************************
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

        --line = reviews[revcnt]; revcnt = revcnt + 1
        line = file_txt:read():lower()
        while line ~= "" and line~=nil do 
            table.insert(sent_txt,line)
            line = file_txt:read():lower()
            --line = reviews[revcnt]; revcnt = revcnt + 1
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
        --line = summaries[sumcnt]; sumcnt = sumcnt + 1
        line = file_smy:read():lower()
        while line ~= "" and line~= nil  do     
            table.insert(sent_smy, line)
            --line = summaries[sumcnt]; sumcnt = sumcnt + 1

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


--*************************************************************
eval = function()
    evalText = io.open('evalText.txt','w')
    evalUserSmy = io.open('target_summary.txt','w')
    evalPredSmy = io.open('predicted_summary.txt','w')
    noPredCnt = 0

    for i = 1,numDocuments/batchSize do
        model:forget()
        print('Evaluated for : ' .. i*batchSize .. 'Reviews')
        local encoderInputs, decoderInputs, decoderTargets = nextBatch()
       
        for batchSize_ind = 1,batchSize do
            str_sent = "" 
            enc_batch_size = encoderInputs:select(2,batchSize_ind)

            --Write review to file
            --for ii = 1,(#enc_batch_size)[1] do   
                --if enc_batch_size[ii] ~= 0 then
                    --local word = Index2Vocab[enc_batch_size[ii]]
                    --str_sent = str_sent .. " " .. word
                --end
            --end 
            --evalText:write(str_sent .. '\n')
            --review_text = str_sent

            -- Create checkwords
            checkword = {}
            if string.match(str_sent:lower(),'interest') then 
                --table.insert(checkword,'interest')
                checkword = 'interest'
            end
            if string.match(str_sent:lower(),'hard') then 
                --table.insert(checkword,'hard')
                checkword = 'hard'
            end
            if string.match(str_sent:lower(),'line') then
                --table.insert(checkword,'line')
                checkword = 'line'
            end 
            if checkword == {} then
                print('No-sense-word-present')
                return
            end

            --Write user summary to file 
            str_sent = "" 
            decoder_batch_size = decoderTargets:select(2,batchSize_ind)

            for ii = 1,(#decoder_batch_size)[1] do
                if decoder_batch_size[ii] ~= 0 then
                    local word = Index2Secab[decoder_batch_size[ii]]
                    str_sent = str_sent .. " " .. word
                end
            end 
            evalUserSmy:write(str_sent .. '\n')
            user_text = str_sent

            --Predicted summary 
            wordIds = model:forward(encoderInputs:select(2,batchSize_ind))

            --Write predicted summary to file 
            str_sent=""
            for _tmp, ind in ipairs(wordIds) do 
                local word = Index2Secab[ind[1]]
                if word ~= nil then 
                    str_sent = str_sent .." ".. word
                else
                    noPredCnt = noPredCnt + 1
                end
            end
            evalPredSmy:write(str_sent .. "\n")
            pred_text = str_sent
            if not string.match(user_text, pred_text) or pred_text == "" then 
                print('____________________________________')
                print("REVIEW : "..review_text)
                print("PREDICTED :"..user_text..":"..pred_text..";")
            end
        end
    end
    evalText:close()
    evalUserSmy:close()
    evalPredSmy:close()
end

--*************************************************************

predict = function(text)
    --getModel(model)
    checkword = {}
    if string.match(text:lower(),'interest') then 
        --table.insert(checkword,'interest')
        checkword = 'interest'
    end
    if string.match(text:lower(),'hard') then 
        --table.insert(checkword,'hard')
        checkword = 'hard'
    end
    if string.match(text:lower(),'line') then
        --table.insert(checkword,'line')
        checkword = 'line'
    end 
    if checkword == {} then
        --print('No-sense-word-present')
        return
    end

    local text = text:split(' ')
    if #text > 100 then
        --print('Too big a sentence')
        return
    end

    local encoderInputs = torch.Tensor(#text)

    for ii=1,#text do
        text[ii] = text[ii]:lower()
        if Vocab[text[ii]] ~= nil then
            encoderInputs[ii] = Vocab[text[ii]]
        else
            encoderInputs[ii] = Vocab['<unknown>']
            print('Unknown words present in the sentence. Aborting : '..text[ii])
            --return
        end
    end

    local wordIds, probabilities, _tmppred = model:eval(encoderInputs,checkword,Secab,Index2Secab,SecabSize)

    print(_tmppred)
    str_sent=""
    for _tmp, ind in ipairs(wordIds) do 
        local word = Index2Secab[ind[1]]
        str_sent = str_sent .." ".. word
    end

    print(str_sent)
    model.encoder:forget()
    model.decoder:forget()
end


--*************************************************************
test = function()
    print('test')

    file_txt = io.open('test_text_words.csv','r')
    file_smy = io.open('test_summary_words.csv','r')


    for i = 1, numDocuments/batchSize do	
        wordIds, decoderTargets = eval()
        model.decoder:forget()
        model.encoder:forget()

        if i*batchSize%100 == 0 then
            print('error for Batch ' .. i .. ' is ' .. " Number Reviews : ".. i*batchSize .. " : ")
        end
    end
    print('Done')
end

eval()
