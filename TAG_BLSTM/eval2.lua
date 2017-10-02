require 'rnn'
require 'optim'
require 'seq2seq'
require 'cutorch'
require 'cunn'
---------------------------------

cmd = torch.CmdLine()
cmd:option('-model','model.t7')
opt = cmd:parse(arg)

cutorch.setDevice(1)
torch.setdefaulttensortype('torch.CudaTensor')
----------------------------------

file_txt=io.open('test_text_words2.csv','r')
file_smy=io.open('test_summary_words2.csv','r')

batchSize = 1
numDocuments = 250

--loaded_model = torch.load('model_tmp')
loaded_model = torch.load(opt.model) 
model = loaded_model.model
Vocab = loaded_model.Vocab
Secab = loaded_model.Secab
Index2Vocab = loaded_model.Index2Vocab 
Index2Secab = loaded_model.Index2Secab
SecabSize = #Index2Secab


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
    return {encoderInputs}, {decoderInputs}, {decoderTargets}, sentBatch_smy
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

    for i = 1,numDocuments/batchSize do
        model:forget()
        --print('Evaluated for : ' .. i*batchSize .. 'Reviews')
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
            
            --evalUserSmy:write(str_sent .. '\n')
            --evalUserSmy:write(Index2Secab[decoder_batch_size[1]] .. '\n')
            evalUserSmy:write(sentBatch_smy[1][2].."\n")
            
            user_text = sentBatch_smy[1][2]

            --Predicted summary 
            probabilities = model:forward(enc_batch_size:resize((#enc_batch_size)[1],1))
            probabilities:resize(SecabSize)
            probabilities, wordIds = probabilities:topk((#probabilities)[1], 1, true, true)
            str_sent = Index2Secab[wordIds[1]]
            
            if str_sent == "" then 
                noPredCnt = noPredCnt + 1
            end

            pred_text = str_sent

            check = 0
            --if pred_text == 'not' and lowermodel ~=nil then 
            if pred_text ~= 'interest6' and lowermodel ~=nil then 
                check = 1
                pred2 =pred_text
                probabilities = lowermodel:forward(enc_batch_size:resize((#enc_batch_size)[1],1))
                probabilities:resize(lowerSecabSize)
                probabilities, wordIds = probabilities:topk((#probabilities)[1], 1, true, true)
                pred_text = lowerIndex2Secab[wordIds[1]] 
            end
            
            if not string.match(user_text, pred_text) then 
                ccnt = ccnt + 1
                if check == 0 then 
                    print("0 :                  :"..pred_text..": : ".. user_text..";")
                else
                    print("1 : INIT: "..pred2.."  :"..pred_text..": : ".. user_text..";")
                end
            end
            evalPredSmy:write(pred_text .. "\n")
        end
    end
    evalText:close()
    evalUserSmy:close()
    evalPredSmy:close()
    os.execute("python quantify_results.py") 
    print("\nacnt = " .. acnt .. " : ccnt = " .. ccnt .. "/" .. numDocuments )
end

--*************************************************************

predict = function(text)
    local text = text:split(' ')
    if #text > 100 then
        print('Too big a sentence')
        return
    end

    local Inputs = torch.Tensor(#text)

    for ii=1,#text do
        text[ii] = text[ii]:lower()
        if Vocab[text[ii]] ~= nil then
            Inputs[ii] = Vocab[text[ii]]
        else
            Inputs[ii] = Vocab['<unknown>']
            print('Unknown words present in the sentence. Aborting : '..text[ii])
            --return
        end
    end

    local wordIds, probabilities, _tmppred = model:forward(Inputs)
    probabilities:resize(5)
    probabilities, wordIds = probabilities:topk((#probabilities)[1], 1, true, true)
    str_sent = Index2Secab[wordIds[1]]

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

    file_txt = io.open('test_text_words2.csv','r')
    file_smy = io.open('test_summary_words2.csv','r')


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
