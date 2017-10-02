-- Based on https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua
local Seq2Seq = torch.class("Seq2Seq")

function Seq2Seq:__init(wordVecSize, vocabSize, secabSize, hiddenSize)
  self.wordVecSize = assert(wordVecSize, "wordVecSize required at arg#1")
  self.vocabSize = assert(vocabSize, "vocabSize required at arg #2")
  self.secabSize = assert(secabSize, "SenseVocabSize required at arg #3")
  self.hiddenSize = assert(hiddenSize, "hiddenSize required at arg #4")
  self:buildModel()
end

function Seq2Seq:buildModel()
  self.encoder = nn.Sequential()
  --self.encoder:add(nn.LookupTableMaskZero(self.vocabSize, self.hiddenSize))
  --self.encoderLSTM = nn.FastLSTM(self.hiddenSize, self.hiddenSize):maskZero(1)
  self.encoderLSTM = nn.FastLSTM(self.wordVecSize, self.hiddenSize):maskZero(1)
  self.encoder:add(nn.Sequencer(self.encoderLSTM))

  --self.encoder:add(nn.Sequencer(nn.FastLSTM(self.hiddenSize,self.hiddenSize)))
  --self.encoder:add(nn.Sequencer(nn.FastLSTM(self.hiddenSize,self.hiddenSize)))

  self.encoder:add(nn.Select(1,-1))

  self.decoder = nn.Sequential()
  --self.decoder:add(nn.LookupTableMaskZero(self.vocabSize, self.hiddenSize))
  self.decoderLSTM = nn.FastLSTM(self.hiddenSize, self.hiddenSize):maskZero(1)
  self.decoder:add(nn.Sequencer(self.decoderLSTM))
  self.decoder:add(nn.Sequencer(nn.MaskZero(nn.Linear(self.hiddenSize, self.secabSize),1)))
  self.decoder:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(),1)))

  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()
end

function Seq2Seq:cuda()
  self.encoder:cuda()
  self.decoder:cuda()

  if self.criterion then
    self.criterion:cuda()
  end
end

function Seq2Seq:float()
  self.encoder:float()
  self.decoder:float()

  if self.criterion then
    self.criterion:float()
  end
end

function Seq2Seq:cl()
  self.encoder:cl()
  self.decoder:cl()

  if self.criterion then
    self.criterion:cl()
  end
end

function Seq2Seq:getParameters()
  return nn.Container():add(self.encoder):add(self.decoder):getParameters()
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function Seq2Seq:forwardConnect(inputSeqLen)
  self.decoderLSTM.userPrevOutput =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevOutput, self.encoderLSTM.outputs[inputSeqLen])
  self.decoderLSTM.userPrevCell =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevCell, self.encoderLSTM.cells[inputSeqLen])
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function Seq2Seq:backwardConnect()
  self.encoderLSTM.userNextGradCell =
    nn.rnn.recursiveCopy(self.encoderLSTM.userNextGradCell, self.decoderLSTM.userGradPrevCell)
  self.encoderLSTM.gradPrevOutput =
    nn.rnn.recursiveCopy(self.encoderLSTM.gradPrevOutput, self.decoderLSTM.userGradPrevOutput)
end

local MAX_OUTPUT_SIZE = 20

function Seq2Seq:eval(input,checkword,Secab,Index2Secab,SecabSize)
  assert(self.goToken, "No goToken specified")
  assert(self.eosToken, "No eosToken specified")

  self.encoder:forward(input)
  self:forwardConnect(input:size(1))

  local predictions = {}
  local probabilities = {}

  -- Forward <go> and all of it's output recursively back to the decoder
  local output = {self.goToken}
  for i = 1, MAX_OUTPUT_SIZE do
    local prediction = self.decoder:forward(torch.Tensor(output))[#output]
    _tmppred = prediction
    -- prediction contains the probabilities for each word IDs.
    -- The index of the probability is the word ID.
    --local prob, wordIds = prediction:topk(5, 1, true, true)

    --local prob, wordIds_all = prediction:topk(SecabSize, 1, true, true)
    local prob, wordIds_all = prediction:topk((#prediction)[1], 1, true, true)

    -- First one is the most likely.
    --next_output = wordIds[1]
    
    --wordIds = {}
    --for j = 1, (#wordIds_all)[1] do
        --tmp = Index2Secab[wordIds_all[j]]
        --if checkword == 'hard' and string.match(tmp,'hard')  then 
            --table.insert(wordIds, wordIds_all[j])
        --elseif checkword == 'interest' and string.match(tmp,'interest') then
            --table.insert(wordIds,wordIds_all[j])
        --elseif checkword == 'line' and (string.match(tmp,'product') or string.match(tmp,'cord') or string.match(tmp,'formation') or string.match(tmp,'division') or string.match(tmp,'phone') or string.match(tmp,'text')) then
            --table.insert(wordIds,wordIds_all[j])
        --elseif wordIds_all[j] == self.eosToken then 
            --table.insert(wordIds,self.eosToken)
        --else
            --table.insert(wordIds,Secab['<unknown>'])
        --end
    --end

    wordIds = wordIds_all
    --print(wordIds)
    next_output = wordIds[1]    

 
    if next_output == self.eosToken then 
        if next(predictions) == nil then 
            _i = 1
            --print(wordIds_all)
            while(wordIds[_i] <= 4 ) do--wordIds[_i] is either of {eosToken, goToken, <unknown>, ""}
                _i =  _i +1
            end 
            wordIds[1] = wordIds[_i]
            wordIds[2] = self.eosToken
            table.insert(predictions,wordIds)
        end
        break
    end

    table.insert(output, next_output)

    -- Terminate on EOS token
    --if next_output == self.eosToken then
        --if next(predictions) == nil  then
            --wordIds[1] = wordIds[2]
            --wordIds[2] = self.eosToken
            --table.insert(predictions,wordIds)
        --end
        --predictions_g = predictions
        --break
    --end

    table.insert(predictions, wordIds)
    table.insert(probabilities, prob)
  end 

  self.decoder:forget()
  self.encoder:forget()

  return predictions, probabilities, _tmppred
end
